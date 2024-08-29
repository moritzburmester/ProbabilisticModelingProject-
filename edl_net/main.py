import torch
import torch.optim as optim
from edl_net import EDLNet_CIFAR10,EDLNet_MNIST, model_training
from auxiliary_functions import (plot_dirichlet_parameters, get_data_loaders, classify_uploaded_image,
                                 plot_first_images_of_each_class, evaluate, test_single_image,
                                 rotating_image_classification, noisy_image_classification)

def main():
    # Configuration parameters
    mode = 'test'  # train or test
    visualize_dir = False  # Set to True to visualize the dirichlet parameters
    selected_classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # Select classes
    num_epochs = 50  # Number of epochs
    num_classes = len(selected_classes)  # Dynamically set the number of classes based on selected_classes
    dataset_name = 'MNIST'  # Change this to 'MNIST','FashionMNIST', or 'CIFAR10'
    save_path = 'pretrained_NN/edl_MNIST_cl_10_ep_50.pth'  # Change filename based on dataset
    img_path = 'data/house.png'  # Path to the uploaded image

    # Loading the data
    test_dataset, train_loader, test_loader = get_data_loaders(dataset_name, batch_size=1000, num_workers=2, root='./data',
                                                               selected_classes=selected_classes)

    # Plot the first image of each class
    plot_first_images_of_each_class(train_loader, num_classes, dataset_name)

    if dataset_name == 'CIFAR10':
        model = EDLNet_CIFAR10(num_classes=num_classes, dropout=True)
    elif dataset_name == 'MNIST' or dataset_name == 'FashionMNIST':
        model = EDLNet_MNIST(num_classes=num_classes, dropout=True)
    else:
        raise ValueError(f"Invalid dataset name: {dataset_name}")

    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.005)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    if mode == 'train':
        model = model_training(
            model,
            train_loader,
            test_loader,
            num_classes=num_classes,
            selected_classes=selected_classes,
            optimizer=optimizer,
            num_epochs=num_epochs,
            save_path=save_path,
            visualize_dir=visualize_dir,
        )
    elif mode == 'test':
        # Evaluate the model using different eval functions
        print("Final Evaluation:")

        # Load the model
        model.load_state_dict(torch.load(save_path), strict=False)
        evaluate(model, test_loader, num_classes, selected_classes)

        # Classify the uploaded image
        print("Classifying uploaded image:")
        if dataset_name == 'CIFAR10':
            input_size = 32
        elif dataset_name == 'MNIST' or dataset_name == 'FashionMNIST':
            input_size = 28
        else:
            raise ValueError(f"Invalid dataset name: {dataset_name}")


        predicted_class = classify_uploaded_image(model, img_path, input_size=input_size,
                                                selected_classes=selected_classes)
        print(f'Predicted class for the uploaded image: {predicted_class}')

        test_single_image(model, img_path, num_classes=num_classes, dataset_name=dataset_name)

        # Filter test dataset for class "1" and visualize the rotation, with a stopper after 3 images
        digit_one_images = [(idx, data) for idx, data in enumerate(test_dataset) if data[1] == 1]
        for idx, (image_idx, (digit_one, _)) in enumerate(digit_one_images):
            print(f'Using image at index {image_idx} from test_dataset for visualization.')
            noisy_image_classification(
                model, img=digit_one, threshold=0.3, num_classes=num_classes,
                selected_classes=selected_classes, plot_dir='noise_classification', file_name=f'noisy_image_{idx}')
            rotating_image_classification(
                model, img=digit_one, threshold=0.3, num_classes=num_classes,
                selected_classes=selected_classes, plot_dir='rotation_classification', file_name=f'rotating_image_{idx}'
            )
            if idx >= 2:  # Stop after processing 3 images
                break

if __name__ == "__main__":
    main()
