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


def test_func(x: float, a: float, b: float, c: float, A:float, phi1: float, phi2:float):
    if x < a or x > c:
        Amp = 0
    else:
        Amp = A
    if x < b:
        phi = phi1
    else:
        phi = phi2
    signal = Amp * np.sin(phi * x)
    return signal

# Define the encoder
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(in_features=256, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=16)


    def forward(self, x):
        x = self.fc1(x)
        x = nn.functional.sigmoid(x) 
        x = self.fc2(x)
        x = nn.functional.sigmoid(x) 
        return x


# Define the decoder
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(in_features=16, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=256)

    def forward(self, x):
        x = self.fc1(x)
        x = nn.functional.sigmoid(x)  
        x = self.fc2(x)
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
def train(model, data_loader, num_epochs=10, learning_rate=0.0001, task=0):
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
        print("Epoch [{}/{}], Loss: {:.4f}".format(epoch + 1, num_epochs, loss.item()))


if __name__ == "__main__":
    plt.style.use("resources/default.mplstyle")
    rand = 7
    np.random.seed(rand)
    torch.manual_seed(rand)
    
    a = -4+ 40 
    b = -4+ 2*40
    c = -4+ 3*40
    A = 1
    phi1 = 1
    phi2 = 0.5

    n_x = 256
    x = np.linspace(0,256,n_x)
    y = [test_func(i,a,b,c,A,phi1,phi2) for i in x]
    # plt.plot(x,y)
    # plt.savefig(f"auto_test_func.png", dpi=100)
    # plt.show()

    n_t = 1000 #1
    step = 0.1
    data = np.ndarray((n_t, n_x))
    print(data.shape)
    for i in range(n_t):
        y = np.array([test_func(j,a+ step*i,b+ step*i,c+ step*i,A,phi1,phi2) for j in x])  /2. +0.5
        # y = np.sin(0.1*x - step*i) /2. +0.5
        data[i] = y
        # print(i, a+ step*i, b+ step*i, c+ step*i, data.shape, y[100], data[i][100])

    time_dim = data.shape[0]
    height_dim = data.shape[1]
    # width_dim = data.shape[2]
    data = data.reshape((time_dim, 1, height_dim))
    print(data.shape)
    # Split data into train and test sets (you can adjust the ratio as needed)
    time_train = int(0.8 * n_t) #// 9*8
    train_data = data[:time_train]
    test_data = data[time_train:]
    print(train_data.shape, test_data.shape)

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

    # # Create an instance of the Convolutional Autoencoder model
    model = ConvAutoencoder()

    # Train the model
    print(test_data.shape)
    train(model, train_loader, num_epochs=100, learning_rate=1e-3)

    torch.save(model,f'model_test_func.pth')
    # model = torch.load(f'model_test_func_lin.pth')

    sample = test_data[0]  # Choose any sample from the test set
    input_tensor = torch.tensor(sample).unsqueeze(0).float()
    reconstructed_output = model(input_tensor).detach().numpy()
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].plot(x,test_data[0].squeeze())
    axes[0].set_title("True")
    axes[1].plot(x,reconstructed_output[0].squeeze())
    axes[1].set_title("Reconstructed")
    axes[0].set_ylim([0, 1])
    axes[1].set_ylim([0, 1])
    fig.tight_layout()
    plt.savefig(f"autoencoder_true_recon_test_func_lin.png", dpi=100)
    
    import winsound
    freq = 500 # Set frequency To 2500 Hertz
    dur = 100 # Set duration To 1000 ms == 1 second
    winsound.Beep(freq, dur)

    print('Done!')