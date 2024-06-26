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

from auxiliary_functions import rotating_image_classification, dempster_shafer, plot_training_metrics,plot_dirichlet_parameters

import torch
import torch.nn as nn
import torch.nn.functional as F

class EDLNet(nn.Module):
    def __init__(self, input_channels=1, num_classes=10, dropout=False, input_size=28, dataset='MNIST'):
        super(EDLNet, self).__init__()
        self.aapl = nn.AdaptiveAvgPool2d((input_size, input_size))  # adaptive layer for variable input sizes
        self.use_dropout = dropout
        self.dataset = dataset

        # Define hyperparameters
        self.num_classes = num_classes
        self.input_channels = input_channels
        self.conv1_out_channels = 8
        self.conv2_out_channels = 16
        self.conv3_out_channels = 128 if dataset == 'CIFAR10' else None
        self.fc1_out_features = 64

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

    #ones = torch.ones([1, num_classes], dtype=torch.float32, device=device)
    ones = torch.ones_like(alpha)  # Ensure `ones` matches the size of `alpha`
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
        selected_classes,
        optimizer,
        num_epochs=25,
        save_path='./cnn.pth',
        visualize_dir=False,
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

    if num_classes != 3:
        visualize_dir = False

    class_mapping = {i: selected_classes[i] for i in range(len(selected_classes))}

    for epoch in range(num_epochs):
        model.train()  # Ensure model is in training mode at the start of each epoch
        print("-" * 120)
        print(f"Epoch {epoch + 1}")
        print("-" * 120)

        epoch_loss = 0.0
        epoch_evidence = 0.0
        epoch_uncertainty = 0.0
        epoch_alpha = []

        k = 50

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

            if k == 50:
                epoch_alpha.append(list(alpha[0].detach().cpu()))
                k = 0
            else:
                k += 1

            labels = torch.eye(num_classes)[labels].float()
            loss = edl_mse_loss(target=labels.float(), epoch_num=epoch, annealing_step=10,
                                num_classes=num_classes, alpha=alpha)

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

        if visualize_dir:
            print(epoch_alpha)
            plot_dirichlet_parameters(epoch_alpha)

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
        test_loader, model, dataclass=7, num_classes=num_classes, threshold=0.2, selected_classes=selected_classes
    )

    # Plotting metrics
    plot_training_metrics(train_evidences, train_uncertainties, num_epochs)


####################################################################################################
#                                       evaluation and testing                                     #
####################################################################################################

def evaluate_model(model_path, test_loader=None, num_classes=3, selected_classes=[7, 8, 9]):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the model
    model = EDLNet(num_classes=num_classes)  # ensure the model is created with the same number of classes
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Model loaded successfully from {model_path}")
    except Exception as e:
        print(f"Error loading the model from {model_path}: {e}")
        return

    model = model.to(device)
    model.eval()

    # Adjust class names based on selected classes
    class_names = [str(cls) for cls in selected_classes]

    print(f"Class names being used: {class_names}")

    # Count occurrences of each class in the test_loader
    class_counts = {cls: 0 for cls in selected_classes}

    for images, labels in test_loader:
        for label in labels:
            class_counts[selected_classes[label.item()]] += 1

    print(f"Class counts in the test set: {class_counts}")

    all_labels = []
    all_predictions = []

    with torch.no_grad():
        n_correct = 0
        n_samples = 0

        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)

            _, predicted = torch.max(outputs, 1)
            n_correct += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            n_samples += labels.size(0)

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the model: {acc:.2f} %')

    # Map predictions and labels back to the original class names
    all_labels = [selected_classes[label] for label in all_labels]
    all_predictions = [selected_classes[pred] for pred in all_predictions]

    # Calculate precision, recall, F1-score
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='weighted', zero_division=1)
    print(f'Precision: {precision:.2f}')
    print(f'Recall: {recall:.2f}')
    print(f'F1-Score: {f1:.2f}')

    # Print classification report
    print('\nClassification Report:')
    print(classification_report(all_labels, all_predictions, target_names=class_names, zero_division=1))

    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)

    # Plot confusion matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()



def single_img_model_evaluate(model, image_path, num_classes, input_channels, input_size,test_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the image
    image = Image.open(image_path).convert('L')
    fig, ax = plt.subplots()
    ax.imshow(image, cmap='gray')
    plt.show()

    # Transform the image
    trans = transforms.Compose([transforms.Resize((input_size, input_size)), transforms.ToTensor()])
    img_tensor = trans(image).unsqueeze(0).to(device)

    # # Plot the transformed image
    img = img_tensor.squeeze().cpu().numpy()  # Squeeze and move to CPU
    plt.imshow(img, cmap='gray')
    plt.show()

    # Make prediction
    with torch.no_grad():
        #outputs = model(images[0].unsqueeze(0).to(device))  # Note: using img_tensor instead of image
        outputs = model(img_tensor)

    print(f'Outputs: {outputs}')

    alpha, dirichlet_strength, uncertainty = dempster_shafer(outputs)
    prob = alpha / torch.sum(alpha, dim=1, keepdim=True)

    _, predicted_class = torch.max(outputs, 1)

    # Convert to numpy for readability
    predicted_class = predicted_class.cpu().item()

    print(f'Alpha: {alpha}')
    print(f'Dirichlet Strength: {dirichlet_strength}')
    print(f'Uncertainty: {uncertainty}')
    print(f'Predicted Class: {predicted_class}')


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