from pathlib import Path
from sympy import reduced

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
# from pytorch_msssim import SSIM

# from comsdk.research import Research
import time

import sys
# sys.path.insert(1, 'C:\\Users\\njuro\\Documents\\esn-studies')

class ConvAutoencoder(nn.Module):
    def __init__(self, latent_dim=8):
        super(ConvAutoencoder, self).__init__()
 
        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7, padding=3),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(512, latent_dim)  # Encoding to a vector of size 8
        )
 
        # Decoder layers
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Unflatten(1, (512, 1, 1)),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(16, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(8, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )
 
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def exponential_smoothing(data, alpha=0.001):
    es = [data[0]]
    for t in range(1, len(data)):
        es.append(alpha * data[t] + (1 - alpha) * es[t - 1])
    return es

# Define the training function
def train(model, data_loader, num_epochs=10, learning_rate=0.0001, task=0):
    loss_fn = nn.BCELoss()
    # ssim_module = SSIM(data_range=255, size_average=True, channel=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    n_batches = len(data_loader)

    writer = np.array([])

    start_time = time.time()  # время начала выполнения

    for epoch in range(num_epochs):
        # for data in (
        #     data_loader, desc=f"Training (epoch {epoch})", total=n_batches
        # ):
        for data in data_loader:
            # img, _ = data
            img = data[0]
            offset1 = np.random.randint(0,128)
            offset2 = np.random.randint(0,128)
            img = torch.roll(img, shifts=(offset1,offset2), dims=(-1,-2))
            
            ###
            img = img.cuda()

            optimizer.zero_grad()
            reconstructed_img = model(img)
            # loss = 1 - ssim_module(reconstructed_img, img)
            
            loss = loss_fn(reconstructed_img, img)
            loss.backward()
            optimizer.step()
        if epoch == num_epochs-1:
            img = data[0]
            # reconstructed_img = model(img)
            reconstructed_img = model(img.cuda())

            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            axes[0].imshow(img[0].squeeze().cpu().detach())
            # axes[0].imshow(img[0].squeeze())
            axes[0].set_title("True")
            axes[1].imshow(reconstructed_img[0].squeeze().cpu().detach().numpy())
            # axes[1].imshow(reconstructed_img[0].squeeze().detach().numpy())
            axes[1].set_title("Reconstructed")
            fig.tight_layout()
            plt.savefig(f"ups_bce_autoen_true_recon_task_{task}_{num_epochs}_lr_{learning_rate}.png", dpi=100)
            # plt.show()
            print()
        writer = np.append(writer, loss.item())
        # scheduler.step(loss)
        if epoch == 0 or np.mod(epoch,10) == 0 or epoch == num_epochs-1:
            print("Epoch [{}/{}], Loss: {:.4f} (lr {})".format(epoch + 1, num_epochs, loss.item(), optimizer.param_groups[0]['lr']))         
    end_time = time.time()  # время окончания выполнения
    execution_time = end_time - start_time  # вычисляем время выполнения
    print(f"Время выполнения {num_epochs} эпох: {execution_time} секунд")

    # print(writer)
    return writer

if __name__ == "__main__":
    # plt.style.use("resources/default.mplstyle")
    rand = 7
    np.random.seed(rand)
    torch.manual_seed(rand)
    
    if len(sys.argv) < 3+1:
        print("You need to run this program like 'python name.py task lrate epo'\n\
              For example python name.py 8 1e-4 50 -- task8, learning rate = 1e-4, 50 epochs")
        exit()
    task = int(sys.argv[1])
    lrate = float(sys.argv[2])
    epo = int(sys.argv[3])
    # task = 8
    fname = {3: "brusselator2DA1.0_B2.1.npz",
            4: "brusselator2DA1.0_B2.5.npz",
            5: "brusselator2DA1.0_B3.0.npz",
            6: "brusselator2DA1.0_B3.5.npz",
            7: "brusselator2DA1.0_B4.0.npz",
            8:'brusselator2DA1.0_B2.5_seed1.npz', 
            9:'brusselator2DA1.0_B2.5_seed2.npz',
            10:'brusselator2DA1.0_B2.5_seed3.npz',
            11:'brusselator2DA1.0_B2.5_seed4.npz'}
    folder = "./drive/MyDrive/Colab Notebooks"
    print(task,folder, fname[task])
    # print(task, fname[task])
    filename = f'{folder}/{fname[task]}'
    data_raw = np.load(filename)
  
    data = data_raw["u"] / np.max(data_raw["u"])
    data = data[:,::2,::2]
    time_dim = data.shape[0]
    height_dim = data.shape[1]
    width_dim = data.shape[2]
    data = data.reshape((time_dim, 1, height_dim, width_dim))
    print(data.shape)
    # Split data into train and test sets (you can adjust the ratio as needed)
    time_train = 2000
    train_data = data[:time_train]
    test_data = data[time_train:]

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
    # model = ConvAutoencoder()
    model = ConvAutoencoder().cuda()

    # Train the model
    print(test_data.shape)
    # lrate = 1e-4
    # epo = 5
    loss_array = train(model, train_loader, num_epochs=epo, learning_rate=lrate, task=task)

    torch.save(model,f'ups_bce_model_roll_task_{task}_epo{epo}_lr_{lrate}.pth')
    # model = torch.load(f'ups_model_task_{task}.pth')

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    ax.plot(np.arange(epo) +1, loss_array, 'b', alpha = 0.6)
    ax.plot(np.arange(epo) +1, exponential_smoothing(loss_array, 0.2), 'b', linewidth= 2)
    ax.set_xlabel(r"$epoch$", fontsize=16)
    ax.set_ylabel(r"$loss$", fontsize=16)

    plt.grid()
    plt.savefig(f"ups_bce_loss_roll_in_train_task_{task}_ep{epo}_lr_{lrate}.png", dpi=100)    

    i=2000-1 # test_data  train_data
    offset = 5 # np.random.randint(0,128)
    # test_data = np.roll(test_data,(offset,offset), (-1,-2)) 
    # sample = test_data[i]  # Choose any sample from the test set
    # input_tensor = torch.tensor(sample).unsqueeze(0).float()

    train_data = np.roll(train_data,(offset,offset), (-1,-2)) 
    sample = train_data[i]  # Choose any sample from the test set
    input_tensor = torch.tensor(sample).float().unsqueeze(0).cuda()

    reconstructed_output = model(input_tensor)
    print(reconstructed_output.shape, train_data[i].shape) #test_data train_data

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(sample.squeeze()) #test_data train_data
    axes[0].set_title("True")
    axes[1].imshow(reconstructed_output[0].cpu().detach().numpy().squeeze())
    axes[1].set_title("Reconstructed")
    fig.tight_layout()
    plt.savefig(f"autoen_roll_in_train_bce_task_{task}_ep{epo}_off_{offset}_train_d_{lrate}.png", dpi=100)
    # plt.savefig(f"autoencoder_true_recon_task_{task}_ep70_roll_i{i}_offset{offset}.png", dpi=100)
    