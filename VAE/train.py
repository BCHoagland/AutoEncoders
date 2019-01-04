import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image

from model import VariationalAutoencoder
from visualize import *

def KL(mu, log_var):
    return -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

# hyperparameters
num_epochs = 100
batch_size = 128
lr = 1e-3

# transformation that normalizes input images
trans = transforms.Compose([
    transforms.ToTensor()
])

# get images from MNIST database
dataset = MNIST('../data', transform=trans, download=True)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# create autoencoder and optimizer for it
autoencoder = VariationalAutoencoder()
optimizer = optim.Adam(autoencoder.parameters(), lr=lr)

# start training
for epoch in range(num_epochs):

    # minibatch optimization with Adam
    for data in dataloader:
        img, _ = data

        # change the images to be 1D
        img = img.view(img.size(0), -1)

        # get output from network
        out, mu, log_var = autoencoder(img)

        # calculate loss and update network
        # loss = binary cross entropy (summed instead of averaged so the BCE loss will dominate any error from random variation) + KL Divergence constraint
        loss = F.binary_cross_entropy(out, img, reduction='sum') + KL(mu, log_var)
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
