from pathlib import Path
from sympy import reduced

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pytorch_msssim import SSIM

from comsdk.research import Research


# Define the encoder
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=8, padding=4)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=8, padding=4)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=8, padding=4)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv4 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=8, padding=4)
        self.pool5 = nn.MaxPool2d(kernel_size=4, stride=4)
        self.conv6 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=8, padding=4)
        self.pool7 = nn.MaxPool2d(kernel_size=4, stride=4)
        self.conv8 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=8, padding=4)
        self.fc9 = nn.Linear(in_features=4, out_features=1)


    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = nn.functional.relu(x)
        x = self.pool3(x)
        x = self.conv4(x)
        x = nn.functional.relu(x)
        x = self.pool5(x)
        x = self.conv6(x)
        x = nn.functional.relu(x)
        x = self.pool7(x)
        x = self.conv8(x)
        x = nn.functional.relu(x)
        x = x.view(x.size()[0],128,4) 
        x = self.fc9(x)
        x = nn.functional.relu(x) 
        return x


# Define the decoder
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.fc0 = nn.Linear(in_features=1, out_features=4)
        self.conv1 = nn.ConvTranspose2d(
            in_channels=128,
            out_channels=64,
            kernel_size=8,
            stride=4,
            padding=4,
            output_padding=1,
        )
        self.conv2 = nn.ConvTranspose2d(
            in_channels=64,
            out_channels=32,
            kernel_size=8,
            stride=4,
            padding=4,
            output_padding=1,
        )
        self.conv3 = nn.ConvTranspose2d(
            in_channels=32,
            out_channels=16,
            kernel_size=8,
            stride=2,
            padding=4,
            output_padding=1,
        )
        self.conv4 = nn.ConvTranspose2d(
            in_channels=16,
            out_channels=8,
            kernel_size=8,
            stride=2,
            padding=4,
            output_padding=1,
        )
        self.conv5 = nn.ConvTranspose2d(
            in_channels=8,
            out_channels=4,
            kernel_size=8,
            stride=2,
            padding=4,
            output_padding=1,
        )
        self.conv6 = nn.ConvTranspose2d(
            in_channels=4, out_channels=1, kernel_size=8, padding=4
        )

    def forward(self, x):
        x = self.fc0(x) 
        x = nn.functional.relu(x)
        x = x.view(x.size()[0],128,2,2) 
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = self.conv3(x)
        x = nn.functional.relu(x)
        x = self.conv4(x)
        x = nn.functional.relu(x)
        x = self.conv5(x)
        x = nn.functional.relu(x)
        x = self.conv6(x)
        x = nn.functional.sigmoid(x)
        return x

# Define the Convolutional Autoencoder
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# Define the training function
def train(model, data_loader, num_epochs=10, learning_rate=0.0001):
    loss_fn = nn.L1Loss()
    ssim_module = SSIM(data_range=255, size_average=True, channel=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    n_batches = len(data_loader)

    for epoch in range(num_epochs):
        for data in tqdm(
            data_loader, desc=f"Training (epoch {epoch})", total=n_batches
        ):
            # img, _ = data
            img = data[0]
            optimizer.zero_grad()
            reconstructed_img = model(img)
            # loss = 1 - ssim_module(reconstructed_img, img)
            loss = loss_fn(reconstructed_img, img)
            loss.backward()
            optimizer.step()
        if epoch == num_epochs-1:
            img = data[0]
            reconstructed_img = model(img)

            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            axes[0].imshow(img[0].squeeze())
            axes[0].set_title("True")
            axes[1].imshow(reconstructed_img[0].squeeze().detach().numpy())
            axes[1].set_title("Reconstructed")
            fig.tight_layout()
            plt.savefig(f"autoencoder_true_reconstructed_{num_epochs}.png", dpi=100)
            plt.show()
            print()
        print("Epoch [{}/{}], Loss: {:.4f}".format(epoch + 1, num_epochs, loss.item()))


if __name__ == "__main__":
    plt.style.use("resources/default.mplstyle")
    rand = 7
    np.random.seed(rand)
    torch.manual_seed(rand)

    task = 2
    res_id = "BRU"
    res = Research.open(res_id)
    task_path = res.get_task_path(task)
    filename = Path(task_path) / "brusselator2DA_1.9B_4.8.npz"
    data_raw = np.load(filename)
    n_pod_components = 2

    # full_dim = np.prod(data_u.shape)
    # full_space_dim = np.prod(data_u.shape[1:])
    # pca = PCA(n_components=n_pod_components)
    # X_u = data_u.reshape(time_dim, full_space_dim)
    # X_v = data_v.reshape(time_dim, full_space_dim)
    # X_u_pca_reduced = pca.fit_transform(X_u)
    # X_v_pca_reduced = pca.fit_transform(X_v)
    # u_v_concat = np.concatenate((X_u_pca_reduced, X_v_pca_reduced), axis=1)

    # Generate data
    # In this example, assume that 'data' is a 4-dimensional tensor of size (num_samples, channels, height, width),
    # where num_samples is the number of samples in the dataset, and each sample is a 2D matrix with real-valued entities
    # (i.e., channels=1)
    # data = generate_data()
    # data_u = data_raw["u"]
    # data_v = data_raw["v"]
    data = data_raw["u"] / np.max(data_raw["u"])
    time_dim = data.shape[0]
    height_dim = data.shape[1]
    width_dim = data.shape[2]
    data = data.reshape((time_dim, 1, height_dim, width_dim))

    # Split data into train and test sets (you can adjust the ratio as needed)
    train_data = data[:600]
    test_data = data[600:]

    # Create dataloaders for the train and test datasets
    batch_size = 8
    train_loader = DataLoader(
        TensorDataset(torch.tensor(train_data).float()),
        batch_size=batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        TensorDataset(torch.tensor(test_data).float()),
        batch_size=batch_size,
        shuffle=False,
    )

    # Create an instance of the Convolutional Autoencoder model
    model = ConvAutoencoder()

    # Train the model
    print(test_data.shape)
    train(model, train_loader, num_epochs=55, learning_rate=1e-3)

    # Use the trained model to reconstruct a sample from the test data
    reduced_data = np.zeros((600, 128)) #512 256 128
    for i in range(600):
        sample = train_data[i]  # Choose any sample from the test set
        input_tensor = torch.tensor(sample).unsqueeze(0).float()
        reduced_data[i, :] = model.encoder(input_tensor).detach().numpy().reshape(-1)
    t = np.arange(0, reduced_data.shape[0], 1)
    red_dim = np.arange(0, reduced_data.shape[1], 1)
    R, T = np.meshgrid(red_dim, t, indexing="ij")
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    cs = ax.contourf(R, T, reduced_data.T, 50)
    cbar = fig.colorbar(cs)
    ax.set_xlabel(r"$c(t)$")
    ax.set_ylabel(r"$t$")
    plt.tight_layout()
    plt.savefig(f"autoencoder_meshgrid.png", dpi=100)
    plt.show()
    print()
    # reconstructed_output = model(input_tensor).detach().numpy()
