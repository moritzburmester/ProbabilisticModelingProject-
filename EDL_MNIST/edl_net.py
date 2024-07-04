import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import time
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, classification_report
import seaborn as sns
import pandas as pd

from auxiliary_functions import (rotating_image_classification, dempster_shafer, plot_training_metrics,
                                 plot_dirichlet_parameters, evaluate_during_training)

class EDLNet(nn.Module):
    def __init__(self, input_channels=1, num_classes=10, dropout=False, input_size=28):
        super(EDLNet, self).__init__()
        self.aapl = nn.AdaptiveAvgPool2d((input_size, input_size))  # adaptive layer for variable input sizes
        self.use_dropout = dropout

        # Define hyperparameters
        self.num_classes = num_classes
        self.input_channels = input_channels
        self.conv1_out_channels = 8    # MNIST 8, FashionMNIST 16
        self.conv2_out_channels = 16   # MNIST 16, FashionMNIST 32
        self.fc1_out_features = 64   # MNIST 64, FashionMNIST 128

        # Define layers
        self.conv1 = nn.Conv2d(self.input_channels, self.conv1_out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(self.conv1_out_channels)
        self.conv2 = nn.Conv2d(self.conv1_out_channels, self.conv2_out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(self.conv2_out_channels)

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

        # Calculate the number of features
        output_size = size * size * self.conv2_out_channels
        return output_size

    def forward(self, x):
        x = self.aapl(x)

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



def edl_mse_loss(target, epoch_num, num_classes, annealing_step, alpha):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Ensure tensors are on the correct device
    y, alpha = target.to(device), alpha.to(device)

    # MSE Loss Calculation
    S = torch.sum(alpha, dim=1, keepdim=True)
    loglikelihood_err = torch.sum((y - (alpha / S)) ** 2, dim=1, keepdim=True)
    loglikelihood_var = torch.sum(
        alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True
    )
    loglikelihood = loglikelihood_err + loglikelihood_var

    # KL Divergence Calculation
    kl_alpha = (alpha - 1) * (1 - y) + 1

    # Ensure `ones` matches the size of `alpha`
    ones = torch.ones_like(alpha)
    sum_kl_alpha = torch.sum(kl_alpha, dim=1, keepdim=True)
    first_term = (
            torch.lgamma(sum_kl_alpha)
            - torch.lgamma(kl_alpha).sum(dim=1, keepdim=True)
            + torch.lgamma(ones).sum(dim=1, keepdim=True)
            - torch.lgamma(ones.sum(dim=1, keepdim=True))
    )
    second_term = (
        (kl_alpha - ones)
        .mul(torch.digamma(kl_alpha) - torch.digamma(sum_kl_alpha))
        .sum(dim=1, keepdim=True)
    )
    kl_div = first_term + second_term

    # Annealing Coefficient
    annealing_coef = torch.min(
        torch.tensor(1.0, dtype=torch.float32),
        torch.tensor(epoch_num / annealing_step, dtype=torch.float32),
    )

    kl_div = annealing_coef * kl_div

    # Total Loss
    loss = torch.mean(loglikelihood + kl_div)

    return loss, torch.mean(loglikelihood_err), torch.mean(loglikelihood_var), torch.mean(kl_div)


def model_training(
        model,
        train_loader,
        test_loader,
        num_classes,
        selected_classes,
        optimizer,
        num_epochs=25,
        save_path='./cnn.pth',
        visualize_dir=False,
):
    since = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    best_model = copy.deepcopy(model.state_dict())
    best_accuracy = 0

    train_evidences = []
    train_uncertainties = []
    train_losses = []
    train_accuracies = []
    test_accuracies = []
    loglikelihood_errors = []
    loglikelihood_variances = []
    kl_divergences = []

    class_mapping = {i: selected_classes[i] for i in range(len(selected_classes))}

    for epoch in range(num_epochs):
        model.train()  # Ensure model is in training mode at the start of each epoch
        print("-" * 120)
        print(f"Epoch {epoch + 1}")
        print("-" * 120)

        epoch_loss = 0.0
        epoch_evidence = 0.0
        epoch_uncertainty = 0.0
        epoch_loglikelihood_error = 0.0
        epoch_loglikelihood_variance = 0.0
        epoch_kl_divergence = 0.0

        for i, (inputs, labels) in enumerate(train_loader):
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

            # apply dempster shafer theory for edl
            alpha, dirichlet_strength, uncertainty = dempster_shafer(outputs)

            labels = torch.eye(num_classes)[labels].float()
            loss, loglikelihood_err, loglikelihood_var, kl_div = edl_mse_loss(
                target=labels.float(), epoch_num=epoch, annealing_step=10,
                num_classes=num_classes, alpha=alpha
            )

            if visualize_dir and i % 100 == 0:
                plot_dirichlet_parameters([list(alpha[0].detach().cpu())])

            epoch_loss += loss.item()
            epoch_loglikelihood_error += loglikelihood_err.item()
            epoch_loglikelihood_variance += loglikelihood_var.item()
            epoch_kl_divergence += kl_div.item()
            epoch_evidence += torch.mean(dirichlet_strength).item()
            epoch_uncertainty += torch.mean(uncertainty).item()

            loss.backward()
            optimizer.step()

        # evaluate the model
        test_accuracy = evaluate_during_training(model, test_loader, num_classes, selected_classes)
        test_accuracies.append(test_accuracy)
        print(f"Epoch {epoch + 1}/{num_epochs}, "
              f"Loss: {(epoch_loss / (i + 1)):.4f}, "
              f"Loglikelihood Error: {(epoch_loglikelihood_error / (i + 1)):.4f}, "
              f"Loglikelihood Variance: {(epoch_loglikelihood_variance / (i + 1)):.4f}, "
              f"KL Divergence: {(epoch_kl_divergence / (i + 1)):.4f}, "
              f"Avg. Evidence: {(epoch_evidence / (i + 1)):.4f}, "
              f"Avg. Uncertainty: {(epoch_uncertainty / (i + 1)):.4f}, "
              f"Train Accuracy: {train_accuracy:.2f}%, "
              f"Test Accuracy: {test_accuracy:.2f}%")

        # save parameters of the best model
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_model = copy.deepcopy(model.state_dict())

        # load best model weights
        model.load_state_dict(best_model)

        # epoch uncertainty
        print(f"Epoch Uncertainty: {epoch_uncertainty / len(train_loader):.4f}")
        train_uncertainties.append(epoch_uncertainty / len(train_loader))

        # epoch evidence
        print(f"Epoch Evidence: {epoch_evidence / len(train_loader):.4f}")
        train_evidences.append(epoch_evidence / len(train_loader))

        # epoch loss
        train_losses.append(epoch_loss / len(train_loader))
        loglikelihood_errors.append(epoch_loglikelihood_error / len(train_loader))
        loglikelihood_variances.append(epoch_loglikelihood_variance / len(train_loader))
        kl_divergences.append(epoch_kl_divergence / len(train_loader))
        print("Epoch Loss: {:.4f}".format(epoch_loss / len(train_loader)))

        if visualize_dir:
            plot_dirichlet_parameters([list(alpha[0].detach().cpu())])

    print("-" * 120)
    print('Finished Training')
    print("-" * 120)
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best Test Accuracy: {best_accuracy:.2f}%')

    # Save the best model
    torch.save(best_model, save_path)
    print("-" * 120)
    print(f'Model saved to {save_path}')

    # Plotting metrics
    plot_training_metrics(train_evidences, train_uncertainties, train_losses, loglikelihood_errors,
                          loglikelihood_variances, kl_divergences, num_epochs)

    return model
