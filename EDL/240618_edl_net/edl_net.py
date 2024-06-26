import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import time

from auxiliary_functions import rotating_image_classification, dempster_shafer, plot_training_metrics

class EDLNet(nn.Module):
    def __init__(self, input_channels=1, num_classes=10, dropout=False, input_size=28):
        super().__init__()
        self.aapl = nn.AdaptiveAvgPool2d((28, 28))  # adaptive layer for variable input sizes
        self.use_dropout = dropout

        # Define hyperparameters
        self.num_classes = num_classes
        self.input_channels = input_channels
        self.conv1_out_channels = 8
        self.conv2_out_channels = 16
        self.fc1_out_features = 64

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
        output_size = size * size * self.conv2_out_channels
        return output_size

    def forward(self, x):
        x = self.aapl(x)

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


def edl_mse_loss(output, target, epoch_num, num_classes, annealing_step, alpha):
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

    ones = torch.ones([1, num_classes], dtype=torch.float32, device=device)
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
    return loss


def model_training(
        model,
        train_loader,
        test_loader,
        num_classes,
        optimizer,
        num_epochs=25,
        save_path='./cnn.pth',
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

        print("-" * 120)
        print(f"Epoch {epoch + 1}")
        print("-" * 120)

        epoch_loss = 0.0
        epoch_evidence = 0.0
        epoch_uncertainty = 0.0

        #k = 0

        for i, (inputs, labels) in enumerate(train_loader):

            # save time if you are not interested in training the model and its results
            #k += 1
            #if k == 5:
                #break

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
                  f"Batch {i + 1}/{len(train_loader)}, "
                  f"Loss: {(epoch_loss / (i + 1)):.4f}, "
                  f"Avg. Evidence: {(epoch_evidence / (i + 1)):.4f}, "
                  f"Avg. Uncertainty: {(epoch_uncertainty / (i + 1)):.4f}, "
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

    print("-" * 120)
    print('Finished Training')
    print("-" * 120)
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best Test Accuracy: {max(test_accuracies):.2f}%')

    # Save the best model
    torch.save(best_model, save_path)
    print("-" * 120)
    print(f'Model saved to {save_path}')

    # Perform rotating image classification to demonstrate uncertainty
    rotating_image_classification(
        test_loader, model, dataclass=1, num_classes=num_classes, threshold=0.5
    )

    # Plotting metrics
    plot_training_metrics(train_evidences, train_uncertainties, num_epochs)