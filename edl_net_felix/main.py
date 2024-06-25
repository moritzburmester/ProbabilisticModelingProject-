import torch
import torch.optim as optim
from edl_net import EDLNet, model_training, evaluate_model
from auxiliary_functions import plot_dirichlet_parameters, get_data_loaders, plot_first_images_of_each_class, image_size_channels


def main():
    # configuration parameters
    selected_classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    num_epochs = 10
    num_classes = len(selected_classes)
    dataset_name = 'MNIST'
    save_path = './edl_mnist_update.pth'

    # loading the data
    train_loader, test_loader = get_data_loaders(dataset_name, batch_size=100, num_workers=0, root='./data',
                                                 selected_classes=selected_classes)

    # plot the first image of each class
    plot_first_images_of_each_class(train_loader, num_classes, dataset_name)

    # get image size and channels
    input_size, input_channels = image_size_channels(train_loader)

    model = EDLNet(input_channels=input_channels, num_classes=num_classes, dropout=False, input_size=input_size)

    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.005)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    model_training(
        model,
        train_loader,
        test_loader,
        num_classes=num_classes,
        optimizer=optimizer,
        num_epochs=num_epochs,
        save_path=save_path,
        visualize_dir=False,
    )

    #evaluate_model(model, './two_test_img.png', num_classes=3, input_channels=1, input_size=28,test_loader=test_loader)
    evaluate_model(save_path,test_loader=test_loader, num_classes=num_classes, selected_classes=selected_classes)

if __name__ == "__main__":
    main()
