import torch
import torch.optim as optim
from edl_net import EDLNet, model_training
from auxiliary_functions import (plot_dirichlet_parameters, get_data_loaders, classify_uploaded_image,
                                 plot_first_images_of_each_class, image_size_channels, evaluate, test_single_image,
                                 rotating_image_classification, noisy_image_classification)

def main():
    # Configuration parameters
    selected_classes = [0, 1, 2,3, 4,5,6,7,8,9]  # Select classes
    num_epochs = 100  # Number of epochs
    num_classes = len(selected_classes)  # Dynamically set the number of classes based on selected_classes
    dataset_name = 'FashionMNIST'  # Focus on MNIST
    save_path = './edl_fashion_mnist.pth'
    image_path = './one.jpg'  # Path to the uploaded image
    img_path = './one.jpg'  # Path to the uploaded image 2nd function

    # Loading the data
    test_dataset, train_loader, test_loader = get_data_loaders(dataset_name, batch_size=1000, num_workers=2, root='./data',
                                                 selected_classes=selected_classes)

    # Plot the first image of each class
    plot_first_images_of_each_class(train_loader, num_classes, dataset_name)

    # Get image size and channels
    input_size, input_channels = image_size_channels(train_loader)

    model = EDLNet(input_channels=input_channels, num_classes=num_classes, dropout=False, input_size=input_size)

    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.005)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # model = model_training(
    #     model,
    #     train_loader,
    #     test_loader,
    #     num_classes=num_classes,
    #     selected_classes=selected_classes,
    #     optimizer=optimizer,
    #     num_epochs=num_epochs,
    #     save_path=save_path,
    #     visualize_dir=False,
    # )


    # Evaluate the model using different eval functions
    print("Final Evaluation:")

    # loaf the model
    model.load_state_dict(torch.load(save_path))
    evaluate(model, test_loader, num_classes, selected_classes)

    # Classify the uploaded image
    print("Classifying uploaded image:")
    predicted_class = classify_uploaded_image(model, image_path, input_size=input_size,
                                              selected_classes=selected_classes)
    print(f'Predicted class for the uploaded image: {predicted_class}')

    test_single_image(model, img_path, num_classes=num_classes)

    # Filter test dataset for class "1" and visualize the rotation, with a stopper after 3 images
    digit_one_images = [(idx, data) for idx, data in enumerate(test_dataset) if data[1] == 1]
    for idx, (image_idx, (digit_one, _)) in enumerate(digit_one_images):
        print(f'Using image at index {image_idx} from test_dataset for visualization.')
        noisy_image_classification(
            model, img=digit_one, threshold=0.3, num_classes=num_classes,
            selected_classes=selected_classes, plot_dir='noise_classification', file_name=f'noisy_image_{idx}')
        rotating_image_classification(
            model, img=digit_one, threshold=0.3, num_classes=num_classes,
            selected_classes=selected_classes, plot_dir='rotation_classification', file_name=f'rotating_image_{idx}')
        if idx >= 2:  # Stop after processing 3 images
            break

if __name__ == "__main__":
    main()
