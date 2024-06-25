import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from torchvision.utils import make_grid
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from main import BNNLeNet, get_data_loaders
from tqdm import tqdm

def plot_uncertainty(model, test_loader, num_samples=50):
    model.eval()
    device = next(model.parameters()).device

    epistemic_uncertainty = []
    aleatoric_uncertainty = []

    with torch.no_grad(), tqdm(test_loader, unit="batch") as tloader:
        for inputs, _ in tloader:
            inputs = inputs.to(device)
            outputs = torch.stack([F.softmax(model(inputs), dim=1) for _ in range(num_samples)], dim=0)
            epistemic_var = torch.var(outputs, dim=0).mean(dim=0).cpu().numpy()
            epistemic_uncertainty.extend(epistemic_var)
            
            if hasattr(model, 'last_noises'):
                aleatoric_var = torch.var(torch.stack(model.last_noises), dim=0).mean(dim=0).cpu().numpy()
                aleatoric_uncertainty.extend(aleatoric_var)
            
            tloader.set_postfix(epistemic_uncertainty=np.mean(epistemic_uncertainty))

    epistemic_uncertainty = np.array(epistemic_uncertainty)
    fig, ax = plt.subplots()
    ax.hist(epistemic_uncertainty, bins=50, alpha=0.5, color='b', label='Epistemic Uncertainty')
    
    if aleatoric_uncertainty:
        aleatoric_uncertainty = np.array(aleatoric_uncertainty)
        ax.hist(aleatoric_uncertainty, bins=50, alpha=0.5, color='r', label='Aleatoric Uncertainty')
    
    ax.set_xlabel('Uncertainty')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Uncertainty')
    ax.legend()
    
    plt.show()

def predict_with_uncertainty(model, inputs, num_samples=50):
    model.eval()
    device = next(model.parameters()).device

    with torch.no_grad(), tqdm(total=len(inputs), unit="batch") as tloader:
        inputs = inputs.to(device)
        outputs = torch.stack([F.softmax(model(inputs), dim=1) for _ in range(num_samples)], dim=0)
        epistemic_var = torch.var(outputs, dim=0).mean(dim=0).cpu().numpy()
        mean_prediction = outputs.mean(dim=0).argmax(dim=1).cpu().numpy()
        tloader.update(len(inputs))
        return mean_prediction, epistemic_var

def display_high_uncertainty_image(model, test_loader, num_samples=50):
    model.eval()
    device = next(model.parameters()).device

    max_uncertainty = -1
    max_uncertainty_image = None
    max_uncertainty_prediction = None

    with torch.no_grad(), tqdm(test_loader, unit="batch") as tloader:
        for inputs, _ in tloader:
            inputs = inputs.to(device)
            outputs = torch.stack([F.softmax(model(inputs), dim=1) for _ in range(num_samples)], dim=0)
            epistemic_var = torch.var(outputs, dim=0).mean(dim=0).cpu().numpy()
            max_batch_uncertainty = np.max(epistemic_var)
            if max_batch_uncertainty > max_uncertainty:
                max_uncertainty = max_batch_uncertainty
                max_uncertainty_index = np.argmax(epistemic_var)
                max_uncertainty_image = inputs[max_uncertainty_index].cpu()
                max_uncertainty_prediction = outputs.mean(dim=0)[max_uncertainty_index].argmax().cpu().numpy()
            
            tloader.set_postfix(max_uncertainty=max_uncertainty)

    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(max_uncertainty_image.squeeze(), cmap='gray')
    plt.axis('off')
    plt.title(f'Predicted Class: {max_uncertainty_prediction}')
    
    plt.subplot(1, 2, 2)
    plt.bar(range(10), outputs.mean(dim=0)[max_uncertainty_index].cpu().numpy())
    plt.xlabel('Classes')
    plt.ylabel('Probability')
    plt.title('Predicted Class Probabilities')
    plt.xticks(range(10))
    plt.tight_layout()
    
    plt.show()

if __name__ == "__main__":
    model = BNNLeNet(input_channels=1, num_classes=10, input_size=28)
    model.load_state_dict(torch.load('./bnn_lenet.pth'))
    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    _, test_loader = get_data_loaders('MNIST', batch_size=32, num_workers=2)

    plot_uncertainty(model, test_loader, num_samples=50)

    inputs, labels = next(iter(test_loader))
    mean_prediction, epistemic_var = predict_with_uncertainty(model, inputs, num_samples=50)
    print(f'Predicted class: {mean_prediction}')
    print(f'Epistemic uncertainty: {epistemic_var}')

    display_high_uncertainty_image(model, test_loader, num_samples=50)
