import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import time
from PIL import Image
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from torchvision import transforms
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, classification_report
import seaborn as sns
import pandas as pd

from auxiliary_functions import rotating_image_classification, dempster_shafer, plot_training_metrics,plot_dirichlet_parameters

import torch
import torch.nn as nn
import torch.nn.functional as F

class EDLNet(nn.Module):
    def __init__(self, input_channels=1, num_classes=10, dropout=False, input_size=28, dataset='MNIST'):
        super(EDLNet, self).__init__()

        self.use_dropout = dropout
        self.dataset = dataset

        # Define hyperparameters
        self.num_classes = num_classes
        self.input_channels = input_channels
        self.conv1_out_channels = 32
        self.conv2_out_channels = 64
        self.conv3_out_channels = 128 
        self.fc1_out_features = 1000

        # Define layers
        self.conv1 = nn.Conv2d(self.input_channels, self.conv1_out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(self.conv1_out_channels)
        self.conv2 = nn.Conv2d(self.conv1_out_channels, self.conv2_out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(self.conv2_out_channels)
        if dataset == 'CIFAR10':
            self.conv3 = nn.Conv2d(self.conv2_out_channels, self.conv3_out_channels, kernel_size=3, padding=1)
            self.bn3 = nn.BatchNorm2d(self.conv3_out_channels)
        
        # Calculate the size of the feature map after conv and pool layers
        conv_output_size = self._get_conv_output_size(input_size)

        # Fully connected layers
        self.fc1 = nn.Linear(conv_output_size, self.fc1_out_features)
        self.fc2 = nn.Linear(self.fc1_out_features, self.num_classes)

    def _get_conv_output_size(self, input_size):
        # Helper function to calculate the size of the output after convolution and pooling
        def conv2d_size_out(size, kernel_size=3, stride=1, padding=1):
            return (size - kernel_size + 2 * padding) // stride + 1

        def maxpool2d_size_out(size, kernel_size=2, stride=2, padding=0):
            return (size - kernel_size + 2 * padding) // stride + 1

        # Calculate size after first conv layer
        size = conv2d_size_out(input_size)
        size = maxpool2d_size_out(size)

        # Calculate size after second conv layer
        size = conv2d_size_out(size)
        size = maxpool2d_size_out(size)

        # Calculate size after third conv layer (if applicable)
        if self.dataset == 'CIFAR10':
            size = conv2d_size_out(size)
            size = maxpool2d_size_out(size)

        # Calculate the number of features
        output_size = size * size * (self.conv3_out_channels if self.dataset == 'CIFAR10' else self.conv2_out_channels)
        return output_size

    def forward(self, x):

        # First convolutional layer
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        # Second convolutional layer
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        if self.dataset == 'CIFAR10':
            # Third convolutional layer (only for CIFAR10)
            x = self.conv3(x)
            x = self.bn3(x)
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

def edl_mse_loss(output, target, epoch_num, num_classes, annealing_step, alpha):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    y, alpha = target.to(device), alpha.to(device)

    S = torch.sum(alpha, dim=1, keepdim=True)
    loglikelihood_err = torch.sum((y - (alpha / S)) ** 2, dim=1, keepdim=True)
    loglikelihood_var = torch.sum(alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True)
    loglikelihood = loglikelihood_err + loglikelihood_var

    kl_alpha = (alpha - 1) * (1 - y) + 1
    ones = torch.ones([1, num_classes], dtype=torch.float32, device=device)
    sum_kl_alpha = torch.sum(kl_alpha, dim=1, keepdim=True)
    first_term = (
        torch.lgamma(sum_kl_alpha) - torch.lgamma(kl_alpha).sum(dim=1, keepdim=True) +
        torch.lgamma(ones).sum(dim=1, keepdim=True) - torch.lgamma(ones.sum(dim=1, keepdim=True))
    )
    second_term = (kl_alpha - ones).mul(torch.digamma(kl_alpha) - torch.digamma(sum_kl_alpha)).sum(dim=1, keepdim=True)
    kl_div = first_term + second_term

    annealing_coef = torch.min(
        torch.tensor(1.0, dtype=torch.float32),
        torch.tensor(epoch_num / annealing_step, dtype=torch.float32),
    )

    kl_div = annealing_coef * kl_div
    loss = torch.mean(loglikelihood + kl_div)
    return loss

def model_training(
        model,
        train_loader,
        test_loader,
        num_classes,
        optimizer,
        num_epochs=25,
        save_path='./cnn.pth',
        visualize_dir=False,
):
    since = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    train_evidences, train_uncertainties, train_losses, train_accuracies, test_accuracies = [], [], [], [], []
    batch_evidences, batch_uncertainties = [], []

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        model.train()
        running_loss, running_corrects, total = 0.0, 0, 0

        k = 50
        epoch_alpha = []

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            corrects = torch.sum(preds == labels.data)

            alpha, dirichlet_strength, uncertainty = dempster_shafer(outputs)
            labels_onehot = torch.eye(num_classes)[labels].to(device)
            loss = edl_mse_loss(outputs, labels_onehot, epoch, num_classes, 10, alpha)

            if k == 50:
                epoch_alpha.append(list(alpha[0].detach().cpu()))
                k = 0
            else:
                k += 1

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += corrects

            batch_evidences.append(torch.mean(dirichlet_strength).item())
            batch_uncertainties.append(torch.mean(uncertainty).item())

        epoch_loss = running_loss / total
        epoch_acc = running_corrects.double() / total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc.item())
        train_evidences.append(torch.mean(dirichlet_strength).item())  # Ensure evidence is collected
        train_uncertainties.append(torch.mean(uncertainty).item())    # Ensure uncertainty is collected

        model.eval()
        test_corrects, test_total = 0, 0

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                test_total += labels.size(0)
                test_corrects += torch.sum(preds == labels.data)

        test_acc = test_corrects.double() / test_total
        test_accuracies.append(test_acc.item())

        if test_acc > best_acc:
            best_acc = test_acc
            best_model_wts = copy.deepcopy(model.state_dict())

        print(f"Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} Test Acc: {test_acc:.4f}")

        if visualize_dir == True:
            plot_dirichlet_parameters(epoch_alpha)

    time_elapsed = time.time() - since
    print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best test Acc: {best_acc:.4f}")

    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), save_path)

    # Ensure the arrays have the same length as the number of epochs
    if len(train_evidences) != num_epochs or len(train_uncertainties) != num_epochs:
        raise ValueError("Length of training metrics arrays does not match the number of epochs.")

    plot_training_metrics(train_evidences, train_uncertainties, num_epochs)

####################################################################################################
#                                       evaluation and testing                                     #
####################################################################################################

def evaluate_model(model, test_loader, num_classes=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()

    all_labels, all_preds = [], []
    with torch.no_grad():
        corrects, total = 0, 0
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            corrects += (preds == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    acc = 100 * corrects / total
    print(f"Accuracy: {acc:.2f}%")

    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
    print(f"Precision: {precision:.2f} Recall: {recall:.2f} F1-Score: {f1:.2f}")


def classify_image(model_path, image_path, input_size=28, num_classes=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the model
    model = EDLNet(input_channels=1 if input_size == 28 else 3, num_classes=num_classes, input_size=input_size)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Model loaded successfully from {model_path}")
    except Exception as e:
        print(f"Error loading the model from {model_path}: {e}")
        return

    model = model.to(device)
    model.eval()

    # Load and preprocess the image
    image = Image.open(image_path).convert('L' if input_size == 28 else 'RGB')
    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)) if input_size == 28 else transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    img_tensor = transform(image).unsqueeze(0).to(device)

    # Plot the image
    plt.imshow(image, cmap='gray' if input_size == 28 else None)
    plt.show()

    # Make prediction
    with torch.no_grad():
        outputs = model(img_tensor)

    print(f'Outputs: {outputs}')

    # Apply Dempster-Shafer theory to get alpha, dirichlet_strength, and uncertainty
    alpha, dirichlet_strength, uncertainty = dempster_shafer(outputs)
    prob = alpha / torch.sum(alpha, dim=1, keepdim=True)

    _, predicted_class = torch.max(outputs, 1)

    # Convert to numpy for readability
    predicted_class = predicted_class.cpu().item()

    print(f'Alpha: {alpha}')
    print(f'Dirichlet Strength: {dirichlet_strength}')
    print(f'Uncertainty: {uncertainty}')
    print(f'Predicted Class: {predicted_class}')

    # Plot the Dirichlet parameters
    plot_dirichlet_parameters(alpha)

    return predicted_class




    



    