import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image

from model import VAE
from visualize import *

# hyperparameters
num_epochs = 50
batch_size = 128
lr = 1e-3

def KL(mu, log_var):
    kl = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    kl /= batch_size * 28 * 28
    return kl

# get images from MNIST database
dataset = MNIST('../data', transform=transforms.ToTensor(), download=True)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# create autoencoder and optimizer for it
vae = VAE()
optimizer = optim.Adam(vae.parameters(), lr=lr)

# start training
for epoch in range(num_epochs):

    # minibatch optimization with Adam
    for data in dataloader:
        img, _ = data

        # change the images to be 1D
        img = img.view(img.size(0), -1)

        # get output from network
        out, mu, log_var = vae(img)

        # calculate loss and update network
        loss = F.binary_cross_entropy(out, img) + KL(mu, log_var)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # save images periodically
    if epoch % 10 == 0:
        img = out.data.view(out.size(0), 1, 28, 28)
        save_image(img, './img/' + str(epoch) + '_epochs.png')

    # plot loss
    update_viz(epoch, loss.item())

# generate new random images
input = torch.randn(96, 10)
out = vae.decode(input)
img = out.data.view(96, 1, 28, 28)
save_image(img, './generated_img/img.png')
