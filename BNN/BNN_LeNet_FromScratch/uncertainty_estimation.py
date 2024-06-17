import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms, datasets
from main import BNNLeNet 
from tqdm import tqdm

norm_mean_std = {
    'MNIST': ([0.5], [0.5]),
    'CIFAR10': ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    'FashionMNIST': ([0.5], [0.5])
}

def load_data(dataset_name, batch_size=32, num_workers=2, root='./data', selected_classes=None):
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

def main():
    num_classes = 10
    dataset_name = 'MNIST'
    batch_size = 32

    # Load data
    train_loader, test_loader = load_data(dataset_name, batch_size=batch_size)

    # Load model
    model = BNNLeNet(input_channels=1, num_classes=num_classes, input_size=28)
    model.load_state_dict(torch.load('./bnn_lenet.pth'))
    model.eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    predictions = []
    uncertainties = []

    # Use tqdm for progress bars
    test_loader = tqdm(test_loader, desc='Predicting', leave=False)
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        batch_predictions = []
        for _ in range(50):  # Perform multiple forward passes for uncertainty estimation
            outputs = model(inputs)
            batch_predictions.append(outputs.detach().cpu().numpy())  # Use detach().cpu().numpy()

        # Concatenate along batch dimension
        batch_predictions = np.concatenate(batch_predictions, axis=0)
        predictions.append(batch_predictions)

        # Calculate predictive uncertainty (e.g., entropy)
        entropy = torch.distributions.Categorical(F.softmax(outputs, dim=1)).entropy()
        uncertainties.append(entropy.detach().cpu().numpy())

    # Concatenate all predictions and uncertainties
    predictions = np.concatenate(predictions, axis=0)
    uncertainties = np.concatenate(uncertainties, axis=0)
    print(uncertainties.shape)
    print(type(uncertainties))
    # Compute predicted labels using mean prediction
    print(predictions)
    print(type(predictions))
    print(predictions.shape)
    mean_predictions = np.mean(predictions, axis=0)  # Take mean across all samples
    print(mean_predictions)
    print(type(mean_predictions))
    print(mean_predictions.shape)
    predicted_labels = np.argmax(mean_predictions, axis=0)  # Find index of maximum value along axis 1
    # Compute accuracy
    accuracy = np.mean(predicted_labels == labels.cpu().numpy())
    print(f"Accuracy: {accuracy * 100:.2f}%")

    # Compute average uncertainty
    avg_uncertainty = np.mean(uncertainties)
    print(f"Average Uncertainty: {avg_uncertainty:.4f}")

if __name__ == "__main__":
    main()
