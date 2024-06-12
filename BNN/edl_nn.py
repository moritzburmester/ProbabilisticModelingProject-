import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim

from torch.utils.data.sampler import SubsetRandomSampler

import numpy as np
import sklearn.datasets
import matplotlib.pyplot as plt
import matplotlib.tri as tri

import copy
import time
import random
import json
import csv
import seaborn as sns
import pandas as pd

############################################################################################################
#                    Data Loader and data manipulation functions                                           #      
############################################################################################################

norm_mean_std = {
        'MNIST': ([0.5], [0.5]),
        'CIFAR10': ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        'FashionMNIST': ([0.5], [0.5])
    }

def get_data_loaders(dataset_name, batch_size=32, num_workers=2, root='./data', selected_classes=None):
    if dataset_name in ['MNIST', 'FashionMNIST']:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(norm_mean_std[dataset_name][0], norm_mean_std[dataset_name][1])
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(norm_mean_std[dataset_name][0], norm_mean_std[dataset_name][1])
        ])

    if dataset_name == 'MNIST':
        train_dataset = torchvision.datasets.MNIST(root=root, train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.MNIST(root=root, train=False, download=True, transform=transform)
    
    elif dataset_name == 'CIFAR10':
        train_dataset = torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=transform)
    
    elif dataset_name == 'FashionMNIST':
        train_dataset = torchvision.datasets.FashionMNIST(root=root, train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.FashionMNIST(root=root, train=False, download=True, transform=transform)
    
    else:
        raise ValueError(f"Dataset {dataset_name} is not supported.")

    if selected_classes is not None:
        def filter_classes(dataset, classes):
            targets = np.array(dataset.targets)
            indices = np.hstack([np.where(targets == cls)[0] for cls in classes])
            dataset.data = dataset.data[indices]
            dataset.targets = targets[indices]
            return dataset
        
        train_dataset = filter_classes(train_dataset, selected_classes)
        test_dataset = filter_classes(test_dataset, selected_classes)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, test_loader

def denormalize(image, mean, std):
    for t, m, s in zip(image, mean, std):
        t.mul_(s).add_(m)
    return image

def plot_first_images_of_each_class(dataloader, num_classes, dataset_name):

    mean, std = norm_mean_std[dataset_name]

    found_classes = set()
    images = []

    for images_batch, labels_batch in dataloader:
        for img, lbl in zip(images_batch, labels_batch):
            if lbl.item() not in found_classes:
                found_classes.add(lbl.item())
                denorm_img = denormalize(img.clone(), mean, std)
                images.append((denorm_img, lbl.item()))
                if len(found_classes) == num_classes:
                    break
        if len(found_classes) == num_classes:
            break
    
    # Plot the images
    fig, axes = plt.subplots(1, num_classes, figsize=(15, 15))
    for img, lbl, ax in zip(images, labels_batch, axes):
        img, lbl = img
        img = img.numpy().transpose(1, 2, 0) if dataset_name == 'CIFAR10' else img.numpy().squeeze()
        ax.imshow(img, cmap='gray' if dataset_name != 'CIFAR10' else None)
        ax.set_title(f'Class: {lbl}')
        ax.axis('off')
    
    plt.show()

def image_size_channels(dataloader):
    images, _ = next(iter(dataloader))
    size = images.shape[-1]
    channels = images.shape[1]
    return size, channels

############################################################################################################
#                                   Neural Network Model                                                   #      
############################################################################################################

class EDLNet(nn.Module):
    def __init__(self, input_channels=1, num_classes=10, dropout=False, input_size=28):
        super().__init__()
        self.use_dropout = dropout
        
        # Define hyperparameters
        self.num_classes = num_classes
        self.input_channels = input_channels
        self.conv1_out_channels = 2
        self.conv2_out_channels = 5
        self.fc1_out_features = 10
        
        # Define layers
        self.conv1 = nn.Conv2d(self.input_channels, self.conv1_out_channels, kernel_size=5)
        self.conv2 = nn.Conv2d(self.conv1_out_channels, self.conv2_out_channels, kernel_size=5)
        
        # Calculate the size of the feature map after conv and pool layers
        conv_output_size = self._get_conv_output_size(input_size)
        
        # Fully connected layers
        self.fc1 = nn.Linear(conv_output_size, self.fc1_out_features)
        self.fc2 = nn.Linear(self.fc1_out_features, self.num_classes)
        
    def _get_conv_output_size(self, input_size):
        # Helper function to calculate the size of the output after convolution and pooling
        def conv2d_size_out(size, kernel_size=5, stride=1, padding=0):
            return (size - kernel_size + 2 * padding) // stride + 1
        
        def maxpool2d_size_out(size, kernel_size=2, stride=2, padding=0):
            return (size - kernel_size + 2 * padding) // stride + 1
        
        # Calculate size after first conv layer
        size = conv2d_size_out(input_size)
        size = maxpool2d_size_out(size)
        
        # Calculate size after second conv layer
        size = conv2d_size_out(size)
        size = maxpool2d_size_out(size)
        
        # Calculate the number of features
        return size * size * self.conv2_out_channels
        
    def forward(self, x):
        # First convolutional layer
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        
        # Second convolutional layer
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        
        # Flatten the tensor
        x = x.view(x.size(0), -1)
        
        # First fully connected layer
        x = self.fc1(x)
        x = F.relu(x)

        # Dropout
        if self.use_dropout:
            x = F.dropout(x, training=self.training)
        
        # Second fully connected layer
        x = self.fc2(x)
        
        return x


############################################################################################################
#                                  Loss Functions                                                          #      
############################################################################################################

def edl_mse_loss(output, target, epoch_num, num_classes, annealing_step, alpha):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Ensure tensors are on the correct device
    y, alpha = target.to(device), alpha.to(device)
    
    # Loglikelihood Loss Calculation
    S = torch.sum(alpha, dim=1, keepdim=True)
    loglikelihood_err = torch.sum((y - (alpha / S)) ** 2, dim=1, keepdim=True)
    loglikelihood_var = torch.sum(alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True)
    loglikelihood = loglikelihood_err + loglikelihood_var
    
    # KL Divergence Calculation
    ones = torch.ones([1, num_classes], dtype=torch.float32, device=device)
    sum_alpha = torch.sum(alpha, dim=1, keepdim=True)
    first_term = (
        torch.lgamma(sum_alpha)
        - torch.lgamma(alpha).sum(dim=1, keepdim=True)
        + torch.lgamma(ones).sum(dim=1, keepdim=True)
        - torch.lgamma(ones.sum(dim=1, keepdim=True))
    )
    second_term = (
        (alpha - ones)
        .mul(torch.digamma(alpha) - torch.digamma(sum_alpha))
        .sum(dim=1, keepdim=True)
    )
    kl_div = first_term + second_term
    
    # Annealing Coefficient
    annealing_coef = torch.min(
        torch.tensor(1.0, dtype=torch.float32),
        torch.tensor(epoch_num / annealing_step, dtype=torch.float32),
    )
    
    kl_alpha = (alpha - 1) * (1 - y) + 1
    kl_div = annealing_coef * kl_div
    
    # Total Loss
    loss = torch.mean(loglikelihood + kl_div)
    return loss

############################################################################################################
#                                  Training and Evaluation Functions                                       #
############################################################################################################

def dempster_shafer(nn_output):
    #evidance
    evidance = F.relu(nn_output)
    # dirichlet distribution concentration parameter a
    concentration = evidance + 1
    # total evidance/dirichlet strength
    dirichlet_strength = torch.sum(concentration, dim=1, keepdim=True)
    # belief masses b
    belief_masses = evidance / dirichlet_strength
    # uncertainty
    uncertainty = 1 - torch.sum(belief_masses, dim=1)

    return concentration, dirichlet_strength, uncertainty

def one_hot_embedding(labels, num_classes=10):
    # Convert to One Hot Encoding
    y = torch.eye(num_classes)
    return y[labels]

def model_training(
    model,
    train_loader,
    test_loader,
    num_classes,
    optimizer,
    num_epochs=25,
    save_path='./cnn.pth',
    json_save_path='./dir_parameters.json'
):
    since = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    best_model = copy.deepcopy(model.state_dict())

    train_evidences = []
    train_uncertainties = []
    train_losses = []
    train_accuracies = []
    test_accuracies = []

    # Initialize lists to store alpha, dirichlet_strength, and uncertainty for every batch and epoch
    batch_data = []

    for epoch in range(num_epochs):

        print("-"*120)
        print(f"Epoch {epoch + 1}")
        print("-"*120)

        epoch_loss = 0.0
        epoch_evidence = 0.0
        epoch_uncertainty = 0.0

        k = 0

        for i, (inputs, labels) in enumerate(train_loader):

            # save time if you are not interested in training the model and its results
            k += 1
            if k == 5:
                break

            # train the model
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)

            # train accuracy
            _, prediction = torch.max(outputs.data, 1)
            total = labels.size(0)
            correct = (prediction == labels).sum().item()
            train_accuracy = 100 * correct / total
            train_accuracies.append(train_accuracy)

            # apply dempster shafer theory
            alpha, dirichlet_strength, uncertainty = dempster_shafer(outputs)

            labels = torch.eye(num_classes)[labels].float()
            loss = edl_mse_loss(outputs, labels.float(), epoch, num_classes, 10, alpha)

            epoch_loss += loss.item()

            epoch_evidence += torch.mean(dirichlet_strength).item()
            epoch_uncertainty += torch.mean(uncertainty).item()

            loss.backward()
            optimizer.step()

            # Save batch data
            batch_data.append({
                'epoch': epoch + 1,
                'batch': i + 1,
                'alpha': alpha.detach().cpu().tolist(),
                'dirichlet_strength': dirichlet_strength.detach().cpu().tolist(),
                'uncertainty': uncertainty.detach().cpu().tolist()
            })

            # evaluate the model
            model.eval()  
            with torch.no_grad():
                correct = 0
                total = 0

                for data in test_loader:
                    inputs, labels = data[0].to(device), data[1].to(device)
                    outputs = model(inputs)
                    _, prediction = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (prediction == labels).sum().item()

            test_accuracy = 100 * correct / total
            test_accuracies.append(test_accuracy)
            print(f"Epoch {epoch + 1}/{num_epochs}, "
                  f"Batch {i+1}/{len(train_loader)}, "
                  f"Loss: {(epoch_loss/(i+1)):.4f}, "
                  f"Avg. Evidance: {(epoch_evidence/(i+1)):.4f}, "
                  f"Avg. Uncertainty: {(epoch_uncertainty/(i+1)):.4f}, "
                  f"Train Accuracy: {train_accuracy:.2f}%, "
                  f"Test Accuracy: {test_accuracy:.2f}%")

        # save parameters of the best model
        if test_accuracy > max(test_accuracies):
            best_model = copy.deepcopy(model.state_dict())

        # epoch uncertainty
        print(f"Epoch Uncertainty: {epoch_uncertainty / len(train_loader):.4f}")
        train_uncertainties.append(epoch_uncertainty / len(train_loader))

        # epoch evidence
        print(f"Epoch Evidence: {epoch_evidence / len(train_loader):.4f}")
        train_evidences.append(epoch_evidence / len(train_loader))

        # epoch loss
        train_losses.append(epoch_loss / len(train_loader))
        print("Epoch Loss: {:.4f}".format(epoch_loss / len(train_loader)))

    print("-"*120)
    print('Finished Training')
    print("-"*120)
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best Test Accuracy: {max(test_accuracies):.2f}%')

    # Save the best model
    torch.save(best_model, save_path)
    print("-"*120)
    print(f'Model saved to {save_path}')

    # Save batch data to JSON file
    with open(json_save_path, 'w') as json_file:
        json.dump(batch_data, json_file)
    print(f'Batch data saved to {json_save_path}')


############################################################################################################
#                                  Plotting Functions                                                      #
############################################################################################################

_corners = np.array([[0, 0], [1, 0], [0.5, 0.75**0.5]])
_AREA = 0.5 * 1 * 0.75**0.5
_triangle = tri.Triangulation(_corners[:, 0], _corners[:, 1])

# For each corner of the triangle, the pair of other corners
_pairs = [_corners[np.roll(range(3), -i)[1:]] for i in range(3)]
# The area of the triangle formed by point xy and another pair or points
tri_area = lambda xy, pair: 0.5 * np.linalg.norm(np.cross(*(pair - xy)))

def xy2bc(xy, tol=1.e-4):
    '''Converts 2D Cartesian coordinates to barycentric.
    Arguments:
        `xy`: A length-2 sequence containing the x and y value.
    '''
    coords = np.array([tri_area(xy, p) for p in _pairs]) / _AREA
    return np.clip(coords, tol, 1.0 - tol)

class Dirichlet(object):
    def __init__(self, alpha):
        '''Creates Dirichlet distribution with parameter `alpha`.'''
        from math import gamma
        from operator import mul
        self._alpha = np.array(alpha)
        self._coef = gamma(np.sum(self._alpha)) / \
                     np.multiply.reduce([gamma(a) for a in self._alpha])
    def pdf(self, x):
        '''Returns pdf value for `x`.'''
        from operator import mul
        return self._coef * np.multiply.reduce([xx ** (aa - 1)
                                                for (xx, aa)in zip(x, self._alpha)])
    def sample(self, N):
        '''Generates a random sample of size `N`.'''
        return np.random.dirichlet(self._alpha, N)

def draw_pdf_contours(dist, border=False, nlevels=200, subdiv=8, **kwargs):
    '''Draws pdf contours over an equilateral triangle (2-simplex).
    Arguments:
        `dist`: A distribution instance with a `pdf` method.
        `border` (bool): If True, the simplex border is drawn.
        `nlevels` (int): Number of contours to draw.
        `subdiv` (int): Number of recursive mesh subdivisions to create.
        kwargs: Keyword args passed on to `plt.triplot`.
    '''

    refiner = tri.UniformTriRefiner(_triangle)
    trimesh = refiner.refine_triangulation(subdiv=subdiv)
    pvals = [dist.pdf(xy2bc(xy)) for xy in zip(trimesh.x, trimesh.y)]

    plt.tricontourf(trimesh, pvals, nlevels, cmap='jet', **kwargs)
    plt.axis('equal')
    plt.xlim(0, 1)
    plt.ylim(0, 0.75**0.5)
    plt.axis('off')
    if border is True:
        plt.triplot(_triangle, linewidth=1)

def plot_points(X, barycentric=True, border=True, **kwargs):
    '''Plots a set of points in the simplex.
    Arguments:
        `X` (ndarray): A 2xN array (if in Cartesian coords) or 3xN array
                       (if in barycentric coords) of points to plot.
        `barycentric` (bool): Indicates if `X` is in barycentric coords.
        `border` (bool): If True, the simplex border is drawn.
        kwargs: Keyword args passed on to `plt.plot`.
    '''
    if barycentric is True:
        X = X.dot(_corners)
    plt.plot(X[:, 0], X[:, 1], 'k.', ms=1, **kwargs)
    plt.axis('equal')
    plt.xlim(0, 1)
    plt.ylim(0, 0.75**0.5)
    plt.axis('off')
    if border is True:
        plt.triplot(_triangle, linewidth=1)

def plot_dirichlet_parameters(alpha_list):
    f = plt.figure(figsize=(8, 6))
    for (i, alpha) in enumerate(alpha_list):
        plt.subplot(2, len(alpha_list), i + 1)
        dist = Dirichlet(alpha)
        draw_pdf_contours(dist)
        title = r'$\alpha$ = (%.3f, %.3f, %.3f)' % tuple(alpha)
        plt.title(title, fontdict={'fontsize': 8})
        plt.subplot(2, len(alpha_list), i + 1 + len(alpha_list))
        plot_points(dist.sample(5000))
    plt.show()

############################################################################################################
#                                  Main Function                                                           #
############################################################################################################

def main():

    # configuraion parameters
    num_epochs = 2
    num_classes = 3
    dataset_name = 'MNIST'

    # test alpha values
    plot_dirichlet_parameters([[1, 1, 16], [1, 13, 14]])

    # loading the data
    train_loader, test_loader = get_data_loaders(dataset_name, batch_size=200, num_workers=2, root='./data', selected_classes=[0, 1, 2])

    # plot the first image of each class
    plot_first_images_of_each_class(train_loader, num_classes, dataset_name)

    # get image size and channels
    input_size, input_channels = image_size_channels(train_loader)

    model = EDLNet(input_channels=input_channels, num_classes=num_classes, dropout=False,input_size=input_size)

    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.005)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    model_training(
        model,
        train_loader,
        test_loader,
        num_classes,
        optimizer,
        num_epochs=num_epochs,
        save_path='./dir_parameters.json',
    )

if __name__ == "__main__":
    main()
