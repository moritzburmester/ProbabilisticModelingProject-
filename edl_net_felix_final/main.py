import torch
import torch.optim as optim
from edl_net import EDLNet, model_training, evaluate_model, rotating_image_classification, classify_image
from auxiliary_functions import (
    plot_dirichlet_parameters, get_data_loader, 
    plot_first_images_of_each_class, image_size_channels,
)

def main():
    # Configuration parameters
    num_epochs = 35
    num_classes = 3
    batch_size=32
    dataset_name = 'CIFAR10'
    save_path = 'edl_mnist_best_model_2.pth'
    test_image_path = 'airplane.jpg'

    # Loading the data
    train_loader, test_loader, class_names = get_data_loader(dataset_name, included_classes=[0, 1, 2], batch_size=batch_size)

    # Plot the first image of each class
    plot_first_images_of_each_class(train_loader, num_classes, dataset_name)

    # Get image size and channels
    input_size, input_channels = image_size_channels(train_loader)

    # Initialize model, optimizer, and device
    model = EDLNet(input_channels=input_channels, num_classes=num_classes, dropout=True, input_size=input_size)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.005)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Train the model
    model_training(
        model,
        train_loader,
        test_loader,
        num_classes,
        optimizer,
        num_epochs=num_epochs,
        save_path=save_path,
        visualize_dir=True,
    )

    # Evaluate the model
    model.load_state_dict(torch.load(save_path))
    evaluate_model(model, test_loader,num_classes)
    rotating_image_classification(test_loader, model, dataclass=2, num_classes=num_classes, threshold=0.7)
    classify_image(save_path, test_image_path, input_size=input_size, num_classes=num_classes)

if __name__ == "__main__":
    main()