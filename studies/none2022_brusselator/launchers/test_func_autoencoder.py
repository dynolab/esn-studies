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
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=4, kernel_size=8, padding=4)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(in_channels=4, out_channels=8, kernel_size=8, padding=4)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv1d(in_channels=8, out_channels=16, kernel_size=8, padding=4)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv4 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=8, padding=4)
        self.fc9 = nn.Linear(in_features=33, out_features=1)


    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.sigmoid(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = nn.functional.sigmoid(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = nn.functional.sigmoid(x)
        x = self.pool3(x)
        
        x = self.conv4(x)
        x = nn.functional.sigmoid(x)

        x = x.view(x.size()[0],x.size()[1],33) 
        x = self.fc9(x)
        x = nn.functional.sigmoid(x) 

        return x


# Define the decoder
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.fc0 = nn.Linear(in_features=1, out_features=33)
        self.conv1 = nn.ConvTranspose1d( in_channels=32, out_channels=16, kernel_size=8, stride=2, 
                                        padding=4, output_padding=1,)
        self.conv2 = nn.ConvTranspose1d( in_channels=16, out_channels=8, kernel_size=8, stride=2, 
                                        padding=4, output_padding=1,)
        self.conv3 = nn.ConvTranspose1d( in_channels=8, out_channels=4, kernel_size=8, stride=2, 
                                        padding=4, output_padding=1,)
        self.conv4 = nn.ConvTranspose1d( in_channels=4, out_channels=1, kernel_size=8, padding=4 )

    def forward(self, x):
        x = self.fc0(x) 
        x = nn.functional.relu(x)
        x = x.view(x.size()[0],x.size()[1],33) 

        x = self.conv1(x)
        x = nn.functional.sigmoid(x)

        x = self.conv2(x)
        x = nn.functional.sigmoid(x)

        x = self.conv3(x)
        x = nn.functional.sigmoid(x)

        x = self.conv4(x)
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
    
    # Generate data
    
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
    # fig, ax = plt.subplots(n_t, 1, figsize=(12, 6))
    for i in range(n_t):
        y = np.array([test_func(j,a+ step*i,b+ step*i,c+ step*i,A,phi1,phi2) for j in x])  /2. +0.5
        # y = np.sin(0.1*x - step*i) /2. +0.5
        data[i] = y
        # print(i, a+ step*i, b+ step*i, c+ step*i, data.shape, y[100], data[i][100])

    # plt.plot(x,data[i-900])
    # plt.plot(x,data[i-600])
    # plt.plot(x,data[i-300])
    # plt.plot(x,data[i])
    #     # plt.plot(x,y) # 
    # plt.show()
    # exit()

    time_dim = data.shape[0]
    height_dim = data.shape[1]
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
    # model = ConvAutoencoder()

    # # Train the model
    # print(test_data.shape)
    # train(model, train_loader, num_epochs=50, learning_rate=1e-3)

    # torch.save(model,f'model_test_func_conv_str2.pth')
    model = torch.load(f'model_test_func_conv_str2_sin.pth')

    sample = test_data[100]  # Choose any sample from the test set  train_data test_data
    input_tensor = torch.tensor(sample).unsqueeze(0).float()
    reconstructed_output = model(input_tensor).detach().numpy()
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].plot(x,test_data[100].squeeze())
    axes[0].set_title("True")
    axes[1].plot(x,reconstructed_output[0].squeeze())
    axes[1].set_title("Reconstructed")
    axes[0].set_ylim([0, 1])
    axes[1].set_ylim([0, 1])
    fig.tight_layout()
    plt.savefig(f"autoencoder_true_recon_test_func_conv.png", dpi=100)
    
    import winsound
    freq = 500 # Set frequency To 2500 Hertz
    dur = 100 # Set duration To 1000 ms == 1 second
    winsound.Beep(freq, dur)

    print('Done!')