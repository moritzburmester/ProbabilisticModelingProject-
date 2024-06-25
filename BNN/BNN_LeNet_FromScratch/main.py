import torch
import torch.nn as nn
import torch.nn.functional as F
import torchbnn as bnn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import copy
import time
from tqdm import tqdm

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

class BNNLeNet(nn.Module):
    def __init__(self, input_channels=1, num_classes=10, input_size=28):
        super(BNNLeNet, self).__init__()
        
        self.conv1 = bnn.BayesConv2d(in_channels=input_channels, out_channels=6, kernel_size=5, prior_sigma=0.1, prior_mu=0.1)
        self.conv2 = bnn.BayesConv2d(in_channels=6, out_channels=16, kernel_size=5, prior_sigma=0.1, prior_mu=0.1)
        self.fc1 = bnn.BayesLinear(in_features=16 * 4 * 4, out_features=120, prior_sigma=0.1, prior_mu=0.1)
        self.fc2 = bnn.BayesLinear(in_features=120, out_features=84, prior_sigma=0.1, prior_mu=0.1)
        self.fc3 = bnn.BayesLinear(in_features=84, out_features=num_classes, prior_sigma=0.1, prior_mu=0.1)
        
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def model_training(model, train_loader, test_loader, optimizer, num_epochs=25, save_path='./bnn_lenet.pth'):
    since = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    best_model = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print('-' * 10)
        
        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        with tqdm(train_loader, unit="batch") as tepoch:
            for inputs, labels in tepoch:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = F.cross_entropy(outputs, labels)
                
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
                tepoch.set_postfix(loss=loss.item())
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        
        print(f'Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model = copy.deepcopy(model.state_dict())
        
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')
    
    model.load_state_dict(best_model)
    torch.save(model.state_dict(), save_path)
    print(f'Model saved to {save_path}')

def main():
    num_epochs = 25
    batch_size = 32
    num_classes = 10
    dataset_name = 'MNIST'
    
    train_loader, test_loader = get_data_loaders(dataset_name, batch_size=batch_size, num_workers=2)
    
    model = BNNLeNet(input_channels=1, num_classes=num_classes, input_size=28)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    model_training(model, train_loader, test_loader, optimizer, num_epochs=num_epochs, save_path='./bnn_lenet.pth')

if __name__ == "__main__":
    main()
