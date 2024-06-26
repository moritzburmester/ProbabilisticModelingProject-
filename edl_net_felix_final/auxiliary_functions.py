import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as nd
import os
from torch.utils.data import DataLoader, Subset, SubsetRandomSampler
import matplotlib.tri as tri
from matplotlib import gridspec
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

############################################################################################################
#                           Data Loader and data manipulation functions                                    #
############################################################################################################

norm_mean_std = {
    'MNIST': ([0.5], [0.5]),
    'CIFAR10': ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    'FashionMNIST': ([0.5], [0.5])
}

def get_data_loader(dataset_name, included_classes, batch_size=32):
    
    if dataset_name == 'MNIST':
        dataset_cls = datasets.MNIST
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        class_names = ['0 - zero', '1 - one', '2 - two', '3 - three', '4 - four', '5 - five', '6 - six', '7 - seven', '8 - eight', '9 - nine']
    elif dataset_name == 'FashionMNIST':
        dataset_cls = datasets.FashionMNIST
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        class_names = ['0 - T-shirt/top', '1 - Trouser', '2 - Pullover', '3 - Dress', '4 - Coat', '5 - Sandal', '6 - Shirt', '7 - Sneaker', '8 - Bag', '9 - Ankle boot']
    elif dataset_name == 'CIFAR10':
        dataset_cls = datasets.CIFAR10
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        class_names = ['0 - airplane', '1 - automobile', '2 - bird', '3 - cat', '4 - deer', '5 - dog', '6 - frog', '7 - horse', '8 - ship', '9 - truck']
    else:
        raise ValueError("Unsupported dataset. Choose from 'MNIST', 'FashionMNIST', 'CIFAR10'.")
    
    # Load the datasets
    train_dataset = dataset_cls(root='./data', train=True, download=True, transform=transform)
    test_dataset = dataset_cls(root='./data', train=False, download=True, transform=transform)
    
    # Function to filter dataset
    def filter_dataset(dataset, included_classes):
        indices = [i for i, (_, label) in enumerate(dataset) if label in included_classes]
        return indices
    
    # Filter datasets
    train_indices = filter_dataset(train_dataset, included_classes)
    test_indices = filter_dataset(test_dataset, included_classes)
    
    # Create Samplers
    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, sampler=test_sampler)
    
    # Filter class names
    filtered_class_names = [class_names[i] for i in included_classes]
    
    return train_loader, test_loader, filtered_class_names

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
#           Transformation of NN Output into evidence and uncertainty  with Dempster Shafer                #
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
#                Visualization of testing an adversarial example displaying the uncertainty                #
############################################################################################################
def rotate_img(x, deg, img_size, channels):
    if channels == 1:
        return nd.rotate(x.reshape(img_size), deg, reshape=False).ravel()
    else:
        rotated = np.zeros_like(x)
        for c in range(channels):
            rotated[c] = nd.rotate(x[c].reshape(img_size), deg, reshape=False)
        return rotated.ravel()

def get_specific_digit(data_loader, digit):
    """Retrieve the first occurrence of the specified digit from the dataset."""
    for imgs, labels in data_loader:
        for img, label in zip(imgs, labels):
            if label.item() == digit:
                return img, label
    return next(iter(data_loader))  # Return the first data point if the specified digit is not found

def rotating_image_classification(dataset, model, dataclass=None, num_classes=10, threshold=0.5, plot_dir='plots'):
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    if dataclass is not None:
        img, label = get_specific_digit(dataset, dataclass)
    else:
        img, label = next(iter(dataset))

    print(f"Using class {label.item()} for classification.")  # Use .item() to convert tensor to a scalar

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    img_size = img.shape[1:]
    channels = img.shape[0]
    Mdeg = 180
    Ndeg = int(Mdeg / 10) + 1
    ldeg = []
    lp = []
    lu = []
    scores = np.zeros((1, num_classes))
    rimgs = np.zeros((img_size[0], img_size[1] * Ndeg, channels))

    for i, deg in enumerate(np.linspace(0, Mdeg, Ndeg)):
        nimg = rotate_img(img.numpy(), deg, img_size, channels).reshape((channels, img_size[0], img_size[1]))
        nimg = np.clip(a=nimg, a_min=0, a_max=1)
        rimgs[:, i * img_size[1]:(i + 1) * img_size[1], :] = nimg.transpose(1, 2, 0)
        nimg = torch.tensor(nimg, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(nimg)

        alpha, diri_strength, uncertainty = dempster_shafer(outputs)
        prob = alpha / torch.sum(alpha, dim=1, keepdim=True)
        p_pred_t = prob.cpu().numpy()
        lu.append(uncertainty.mean().cpu().numpy())

        scores += p_pred_t >= threshold
        ldeg.append(deg)
        lp.append(p_pred_t[0])

    labels = np.arange(num_classes)[scores[0].astype(bool)]
    lp = np.array(lp)[:, labels]
    c = ['black', 'blue', 'brown', 'red', 'purple', 'cyan']
    marker = ['s', '^', 'o'] * 2
    labels = labels.tolist()

    # Create a single figure with two subplots using GridSpec
    fig = plt.figure(figsize=(10, 8))
    gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1], hspace=0.05)

    # Add a title to the figure
    fig.suptitle(f"Classification and uncertainty for rotated image of class {label.item()}", fontsize=14)

    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])

    # Plot the classification probabilities
    for i in range(len(labels)):
        ax0.plot(ldeg, lp[:, i], marker=marker[i], c=c[i])

    if len(lu) > 0:
        labels += ['uncertainty']
        ax0.plot(ldeg, lu, marker='<', c='red')

    ax0.legend(labels)
    ax0.set_xlim([0, Mdeg])
    ax0.set_xlabel('Rotation Degree')
    ax0.set_ylabel('Classification Probability')

    # Plot the rotated images underneath
    ax1.imshow(1 - rimgs, cmap='gray', aspect='equal')
    ax1.axis('off')

    # Save the combined plot
    plt.savefig(f'{plot_dir}/testing_rotation.png')
    plt.show()


#plotting the development of uncertainty and evidence
def plot_training_metrics(train_evidences, train_uncertainties, num_epochs):
    epochs = range(1, num_epochs + 1)

    fig, ax1 = plt.subplots(figsize=(12, 6))

    color = 'tab:blue'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Evidence (Dirichlet Strength)', color=color)
    ax1.plot(epochs, train_evidences, label='Evidence (Dirichlet Strength)', color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:red'
    ax2.set_ylabel('Uncertainty', color=color)  # we already handled the x-label with ax1
    ax2.plot(epochs, train_uncertainties, label='Uncertainty', color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.legend(loc='upper right')

    fig.tight_layout()  # Adjusts the padding to fit the elements in the figure area
    fig.subplots_adjust(top=0.9)  # Adjust the top of the subplot to fit the title
    plt.title('Evidence (Dirichlet Strength) and Uncertainty during Training')
    plt.show()

