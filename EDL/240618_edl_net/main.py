import torch
import torch.optim as optim
from edl_net import EDLNet, model_training
from auxiliary_functions import plot_dirichlet_parameters, get_data_loaders, plot_first_images_of_each_class, image_size_channels

def main():
    # configuration parameters
    num_epochs = 2
    num_classes = 3
    dataset_name = 'MNIST'

    # test alpha values
    #plot_dirichlet_parameters([[1, 1, 16], [1, 13, 14]])

    # loading the data
    train_loader, test_loader = get_data_loaders(dataset_name, batch_size=200, num_workers=0, root='./data',
                                                 selected_classes=[0, 1, 2])

    # plot the first image of each class
    #plot_first_images_of_each_class(train_loader, num_classes, dataset_name)

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
        num_classes,
        optimizer,
        num_epochs=num_epochs,
        save_path='./cnn.pth',
    )

if __name__ == "__main__":
    main()
