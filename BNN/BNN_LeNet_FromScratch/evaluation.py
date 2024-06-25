import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import time
from tqdm import tqdm
from main import BNNLeNet, get_data_loaders, model_training

def evaluate_model(model, test_loader):
    model.eval()
    device = next(model.parameters()).device

    running_corrects = 0
    running_loss = 0.0
    uncertainties = []

    with torch.no_grad(), tqdm(test_loader, unit="batch") as tloader:
        for inputs, labels in tloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = F.cross_entropy(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            
            # Compute uncertainty
            num_samples = 50
            sampled_outputs = torch.stack([F.softmax(model(inputs), dim=1) for _ in range(num_samples)], dim=0)
            epistemic_var = torch.var(sampled_outputs, dim=0).mean(dim=0).cpu().numpy()
            uncertainties.extend(epistemic_var)
            
            tloader.set_postfix(loss=loss.item(), accuracy=running_corrects.double() / len(test_loader.dataset))

    final_loss = running_loss / len(test_loader.dataset)
    final_accuracy = running_corrects.double() / len(test_loader.dataset)
    average_uncertainty = np.mean(uncertainties)
    
    return final_loss, final_accuracy.item(), average_uncertainty

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
    
    start_time = time.time()
    model_training(model, train_loader, test_loader, optimizer, num_epochs=num_epochs, save_path='./bnn_lenet.pth')
    end_time = time.time()
    
    runtime = end_time - start_time
    final_loss, final_accuracy, average_uncertainty = evaluate_model(model, test_loader)
    
    data = {
        'Final Accuracy': [final_accuracy],
        'Runtime (s)': [runtime],
        'Final Loss': [final_loss],
        'Average Uncertainty': [average_uncertainty]
    }
    
    df = pd.DataFrame(data)
    df.to_csv('evaluation_metrics.csv', index=False)
    print(df)

if __name__ == "__main__":
    main()
