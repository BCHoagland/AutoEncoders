import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image

from model import Autoencoder
from visualize import *

# hyperparameters
num_epochs = 100
batch_size = 128
lr = 1e-3

# get images from MNIST database
dataset = MNIST('../data', transform=transforms.ToTensor(), download=True)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# create autoencoder and optimizer for it
autoencoder = Autoencoder()
optimizer = optim.Adam(autoencoder.parameters(), lr=lr)

# start training
for epoch in range(num_epochs):

    # minibatch optimization with Adam
    for data in dataloader:
        img, _ = data

        # change the images to be 1D
        img = img.view(img.size(0), -1)

        # get output from network
        out = autoencoder(img)

        # calculate MSE loss and update network
        loss = torch.pow(img - out, 2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # save images periodically
    if epoch % 10 == 0:
        img = out.data * 0.5 + 0.5
        img = img.view(out.size(0), 1, 28, 28)
        save_image(img, './img/' + str(epoch) + '_epochs.png')

    # plot loss
    update_viz(epoch, loss.item())
