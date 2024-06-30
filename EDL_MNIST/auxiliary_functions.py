from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, classification_report
import pandas as pd
import seaborn as sns
from PIL import Image
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as nd
import time
import os
import matplotlib.tri as tri
import matplotlib.gridspec as gridspec
from scipy.ndimage import rotate as nd_rotate
from scipy.ndimage import gaussian_filter


############################################################################################################
#                           Data Loader and data manipulation functions                                    #
############################################################################################################

norm_mean_std = {
    'MNIST': ([0.5], [0.5]),
    'FashionMNIST': ([0.5], [0.5])
}

def get_data_loaders(dataset_name, batch_size=32, num_workers=0, root='./data', selected_classes=None):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(norm_mean_std[dataset_name][0], norm_mean_std[dataset_name][1])
    ])

    if dataset_name == 'MNIST':
        train_dataset = torchvision.datasets.MNIST(root=root, train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.MNIST(root=root, train=False, download=True, transform=transform)
    elif dataset_name == 'FashionMNIST':
        train_dataset = torchvision.datasets.FashionMNIST(root=root, train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.FashionMNIST(root=root, train=False, download=True, transform=transform)
    else:
        raise ValueError(f"Dataset {dataset_name} is not supported.")

    if selected_classes is not None:
        new_labels = {cls: idx for idx, cls in enumerate(selected_classes)}

        def filter_and_remap_classes(dataset, classes, new_labels):
            targets = np.array(dataset.targets)
            indices = np.hstack([np.where(targets == cls)[0] for cls in classes])
            dataset.data = dataset.data[indices]
            dataset.targets = np.array([new_labels[cls] for cls in targets[indices]])
            return dataset

        train_dataset = filter_and_remap_classes(train_dataset, selected_classes, new_labels)
        test_dataset = filter_and_remap_classes(test_dataset, selected_classes, new_labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return test_dataset, train_loader, test_loader

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
        img = img.numpy().squeeze()
        ax.imshow(img, cmap='gray')
        ax.set_title(f'Class: {lbl}')
        ax.axis('off')

    plt.show()

def image_size_channels(dataloader):
    images, _ = next(iter(dataloader))
    size = images.shape[-1]
    channels = images.shape[1]
    return size, channels


############################################################################################################
#                              Visualization of Dirichlet Distribution                                     #
############################################################################################################

_corners = np.array([[0, 0], [1, 0], [0.5, 0.75 ** 0.5]])
_AREA = 0.5 * 1 * 0.75 ** 0.5
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
                                                for (xx, aa) in zip(x, self._alpha)])

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
    plt.ylim(0, 0.75 ** 0.5)
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
    plt.ylim(0, 0.75 ** 0.5)
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
#           Transformation of NN Output into evidence and uncertainty with Dempster Shafer                 #
############################################################################################################
def dempster_shafer(nn_output):
    # evidence
    evidence = F.relu(nn_output)
    # dirichlet distribution concentration parameter a
    concentration = evidence + 1
    # total evidence/dirichlet strength
    dirichlet_strength = torch.sum(concentration, dim=1, keepdim=True)
    # belief masses b
    belief_masses = evidence / dirichlet_strength
    # uncertainty
    uncertainty = 1 - torch.sum(belief_masses, dim=1)

    return concentration, dirichlet_strength, uncertainty


############################################################################################################
#                                       Plotting training metrics                                          #
############################################################################################################
def plot_training_metrics(evidences, uncertainties, losses, loglikelihood_errors, loglikelihood_variances,
                          kl_divergences, num_epochs):
    epochs = range(1, num_epochs + 1)

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 3, 1)
    plt.plot(epochs, evidences, label='Evidence')
    plt.xlabel('Epochs')
    plt.ylabel('Evidence')
    plt.legend()

    plt.subplot(2, 3, 2)
    plt.plot(epochs, uncertainties, label='Uncertainty')
    plt.xlabel('Epochs')
    plt.ylabel('Uncertainty')
    plt.legend()

    plt.subplot(2, 3, 3)
    plt.plot(epochs, losses, label='Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(2, 3, 4)
    plt.plot(epochs, loglikelihood_errors, label='Loglikelihood Error')
    plt.xlabel('Epochs')
    plt.ylabel('Loglikelihood Error')
    plt.legend()

    plt.subplot(2, 3, 5)
    plt.plot(epochs, loglikelihood_variances, label='Loglikelihood Variance')
    plt.xlabel('Epochs')
    plt.ylabel('Loglikelihood Variance')
    plt.legend()

    plt.subplot(2, 3, 6)
    plt.plot(epochs, kl_divergences, label='KL Divergence')
    plt.xlabel('Epochs')
    plt.ylabel('KL Divergence')
    plt.legend()

    plt.tight_layout()
    plt.show()


############################################################################################################
#                                          evaluation function                                             #
############################################################################################################
def evaluate_during_training(model, data_loader, num_classes, selected_classes):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        n_correct = 0
        n_samples = 0

        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)

            _, predicted = torch.max(outputs, 1)
            n_correct += (predicted == labels).sum().item()
            n_samples += labels.size(0)

    acc = 100.0 * n_correct / n_samples

    return acc

def evaluate(model, data_loader, num_classes, selected_classes, save_dir='plots'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    all_labels = []
    all_predictions = []

    with torch.no_grad():
        n_correct = 0
        n_samples = 0

        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)

            _, predicted = torch.max(outputs, 1)
            n_correct += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            n_samples += labels.size(0)

    acc = 100.0 * n_correct / n_samples

    # Map predictions and labels back to the original class names
    all_labels = [selected_classes[label] for label in all_labels]
    all_predictions = [selected_classes[pred] for pred in all_predictions]

    # Calculate precision, recall, F1-score
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='weighted', zero_division=1)

    # Print classification report
    print('\nClassification Report:')
    print(classification_report(all_labels, all_predictions, target_names=[str(cls) for cls in selected_classes], zero_division=1))

    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    cm_df = pd.DataFrame(cm, index=[str(cls) for cls in selected_classes], columns=[str(cls) for cls in selected_classes])

    # Plot confusion matrix with increased font size for annotations
    plt.figure(figsize=(12, 10))
    ax = sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', cbar_kws={'label': 'Scale'}, annot_kws={"size": 12})  # Adjust font size and color map
    plt.xlabel('Predicted', fontsize=16)
    plt.ylabel('Actual', fontsize=16)
    plt.title('Confusion Matrix', fontsize=20)

    # Save the confusion matrix plot
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))

    # Show the plot
    plt.show()

    return acc, precision, recall, f1



############################################################################################################
#                Visualization of testing an adversarial example displaying the uncertainty                #
############################################################################################################

def test_single_image(model, img_path, num_classes=10):
    start_time = time.time()  # Start time tracking

    img = Image.open(img_path).convert("L")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trans = transforms.Compose(
        [transforms.Resize((28, 28)), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    img_tensor = trans(img).unsqueeze(0).to(device)  # Ensure normalization and move to device

    model = model.to(device)
    model.eval()
    with torch.no_grad():
        output = model(img_tensor)

    alpha, diri_strength, uncertainty = dempster_shafer(output)
    _, preds = torch.max(output, 1)
    prob = alpha / torch.sum(alpha, dim=1, keepdim=True)

    output = output.flatten()
    prob = prob.flatten()
    preds = preds.flatten()

    labels = np.arange(num_classes)
    fig, axs = plt.subplots(1, 2, gridspec_kw={"width_ratios": [1, 3]})

    plt.title("Classified as: {}, Uncertainty: {}".format(preds[0].item(), uncertainty.item()))

    axs[0].set_title("Image")
    axs[0].imshow(img, cmap="gray")
    axs[0].axis("off")

    axs[1].bar(labels, prob.cpu().detach().numpy(), width=0.5)
    axs[1].set_xlim([0, 9])
    axs[1].set_ylim([0, 1])
    axs[1].set_xticks(np.arange(10))
    axs[1].set_xlabel("Classes")
    axs[1].set_ylabel("Classification Probability")

    fig.tight_layout()

    plt.savefig("./plots/{}".format(os.path.basename(img_path)))

    plt.show()

    end_time = time.time()  # End time tracking
    runtime = end_time - start_time  # Calculate runtime

    print(f'Runtime: {runtime:.4f} seconds')  # Print runtime


def rotate_img(x, deg):
    return nd_rotate(x.reshape(28, 28), deg, reshape=False).ravel()





def classify_uploaded_image(model, image_path, input_size=28, selected_classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Record the start time
    start_time = time.time()

    # Load and preprocess the image
    image = Image.open(image_path).convert('L' if input_size == 28 else 'RGB')
    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)) if input_size == 28 else transforms.Normalize(
            (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    img_tensor = transform(image).unsqueeze(0).to(device)

    # Plot the image
    plt.imshow(image, cmap='gray' if input_size == 28 else None)
    plt.title('Uploaded Image')
    plt.show()

    # Make prediction
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(img_tensor)

    # Apply Dempster-Shafer theory to get alpha, dirichlet_strength, and uncertainty
    alpha, dirichlet_strength, uncertainty = dempster_shafer(outputs)
    prob = alpha / torch.sum(alpha, dim=1, keepdim=True)

    _, predicted_class = torch.max(outputs, 1)

    # Convert to numpy for readability
    predicted_class = predicted_class.cpu().item()

    # Record the end time
    end_time = time.time()

    # Calculate the runtime
    runtime = end_time - start_time

    print(f'Alpha: {alpha}')
    print(f'Dirichlet Strength: {dirichlet_strength}')
    print(f'Uncertainty: {uncertainty}')
    print(f'Predicted Class: {selected_classes[predicted_class]}')
    print(f'Predicted Probability: {prob[0][predicted_class].cpu().item()}')
    print(f'Runtime: {runtime:.4f} seconds')

    return selected_classes[predicted_class]


def unnormalize(img, mean, std):
    """Unnormalize the image for visualization."""
    img = img * std + mean
    return img

def rotate_img(x, deg):
    """Rotate image by a specified degree."""
    return nd_rotate(x.reshape(28, 28), deg, reshape=False).ravel()

def rotating_image_classification(model, img, threshold=0.5, num_classes=10, selected_classes=None,
                                  plot_dir='rotation_classification', file_name='rotating_image'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    Mdeg = 180
    Ndeg = int(Mdeg / 10) + 1
    ldeg = []
    lp = []
    lu = []
    classifications = []

    scores = np.zeros((1, num_classes))
    rimgs = np.zeros((28, 28 * Ndeg))

    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    for i, deg in enumerate(np.linspace(0, Mdeg, Ndeg)):
        nimg = rotate_img(np.array(img), deg).reshape(28, 28)
        nimg = np.clip(a=nimg, a_min=0, a_max=1)
        rimgs[:, i * 28: (i + 1) * 28] = nimg
        img_tensor = trans(nimg).unsqueeze(0).to(device)  # Ensure normalization and move to device

        model = model.to(device)
        model.eval()
        with torch.no_grad():
            output = model(img_tensor)
        alpha, diri_strength, uncertainty = dempster_shafer(output)
        prob = alpha / torch.sum(alpha, dim=1, keepdim=True)
        uncertainty = num_classes / torch.sum(alpha, dim=1, keepdim=True)
        _, preds = torch.max(output, 1)
        classifications.append(selected_classes[preds[0].item()])
        lu.append(uncertainty.mean().detach().cpu().numpy())

        scores += prob.detach().cpu().numpy() >= threshold
        ldeg.append(deg)
        lp.append(prob.cpu().numpy())

    labels = np.arange(num_classes)[scores[0].astype(bool)]

    # Convert lp to numpy array and squeeze the unnecessary dimension
    lp_array = np.squeeze(np.array(lp), axis=1)

    lp = lp_array[:, labels]

    c = ['black', 'blue', 'brown', 'purple', 'cyan', 'red', 'green']
    marker = ['s', '^', 'o'] * 2
    labels = [selected_classes[label] for label in labels.tolist()]

    fig = plt.figure(figsize=(10, 8))
    gs = gridspec.GridSpec(3, 1, height_ratios=[2, 1, 0.2], hspace=0.05)
    fig.suptitle(f"Classification and uncertainty for rotated image of class {selected_classes[classifications[0]]}", fontsize=14)

    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])
    ax2 = plt.subplot(gs[2])

    for i in range(len(labels)):
        ax0.plot(ldeg, lp[:, i], marker=marker[i], c=c[i])

    if len(lu) > 0:
        labels += ['uncertainty']
        ax0.plot(ldeg, lu, marker='<', c='red')

    ax0.legend(labels)
    ax0.set_xlim([0, Mdeg])
    ax0.set_xlabel('Rotation Degree')
    ax0.set_ylabel('Classification Probability')

    ax1.imshow(1 - rimgs, cmap='gray', aspect='equal')
    ax1.axis('off')

    ax2.axis('off')
    table_data = [classifications]
    table = ax2.table(cellText=table_data, cellLoc='center', loc='center', edges='open')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.5)
    ax2.set_title('Classifications')

    for key, cell in table.get_celld().items():
        cell.set_linewidth(1)

    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    plt.savefig(os.path.join(plot_dir, f'{file_name}.png'))
    plt.show()


############################################################################################################
#                                         Gaussian                                           #
############################################################################################################

def add_gaussian_noise(image, std_dev):
    noise = np.random.normal(0, std_dev, image.shape)
    noisy_image = image + noise
    return np.clip(noisy_image, 0, 1)

def noisy_image_classification(model, img, threshold=0.5, num_classes=10, selected_classes=None,
                               plot_dir='noise_classification', file_name='noisy_image'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    max_std_dev = 1.0
    num_steps = 20
    l_std_dev = []
    lp = []
    lu = []
    classifications = []

    scores = np.zeros((1, num_classes))
    nimgs = np.zeros((28, 28 * num_steps))

    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    for i, std_dev in enumerate(np.linspace(0, max_std_dev, num_steps)):
        nimg = add_gaussian_noise(np.array(img), std_dev).reshape(28, 28)
        nimg = np.clip(a=nimg, a_min=0, a_max=1)
        nimgs[:, i * 28: (i + 1) * 28] = nimg
        img_tensor = trans(nimg).unsqueeze(0).to(device)  # Ensure normalization and move to device

        img_tensor = img_tensor.float()  # Convert to float32

        model = model.to(device)
        model.eval()
        with torch.no_grad():
            output = model(img_tensor)
        alpha, diri_strength, uncertainty = dempster_shafer(output)
        prob = alpha / torch.sum(alpha, dim=1, keepdim=True)
        uncertainty = num_classes / torch.sum(alpha, dim=1, keepdim=True)
        _, preds = torch.max(output, 1)
        classifications.append(selected_classes[preds[0].item()])
        lu.append(uncertainty.mean().detach().cpu().numpy())

        scores += prob.detach().cpu().numpy() >= threshold
        l_std_dev.append(std_dev)
        lp.append(prob.cpu().numpy())

    labels = np.arange(num_classes)[scores[0].astype(bool)]

    # Convert lp to numpy array and squeeze the unnecessary dimension
    lp_array = np.squeeze(np.array(lp), axis=1)

    lp = lp_array[:, labels]

    c = ['black', 'blue', 'brown', 'purple', 'cyan', 'red', 'green']
    marker = ['s', '^', 'o'] * 2
    labels = [selected_classes[label] for label in labels.tolist()]

    fig = plt.figure(figsize=(10, 8))
    gs = plt.GridSpec(3, 1, height_ratios=[2, 1, 0.2], hspace=0.05)
    fig.suptitle(f"Classification and uncertainty for noisy image of class {selected_classes[classifications[0]]}", fontsize=14)

    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])
    ax2 = plt.subplot(gs[2])

    for i in range(len(labels)):
        ax0.plot(l_std_dev, lp[:, i], marker=marker[i], c=c[i])

    if len(lu) > 0:
        labels += ['uncertainty']
        ax0.plot(l_std_dev, lu, marker='<', c='red')

    ax0.legend(labels)
    ax0.set_xlim([0, max_std_dev])
    ax0.set_xlabel('Gaussian Noise Std Dev')
    ax0.set_ylabel('Classification Probability')

    ax1.imshow(1 - nimgs, cmap='gray', aspect='equal')
    ax1.axis('off')

    ax2.axis('off')
    table_data = [classifications]
    table = ax2.table(cellText=table_data, cellLoc='center', loc='center', edges='open')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.5)
    ax2.set_title('Classifications')

    for key, cell in table.get_celld().items():
        cell.set_linewidth(1)

    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    plt.savefig(os.path.join(plot_dir, f'{file_name}.png'))
    plt.show()
