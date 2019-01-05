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
num_epochs = 100
batch_size = 128
lr = 1e-3
beta = 4

def KL(mu, log_var):
    kl = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    kl /= batch_size * 28 * 28
    return kl

# get images from MNIST database
dataset = MNIST('../data', transform=transforms.ToTensor(), download=True)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# create ß-vae and its optimizer
beta_vae = VAE()
beta_vae_optimizer = optim.Adam(beta_vae.parameters(), lr=lr)

# start training
for epoch in range(num_epochs):

    # minibatch optimization with Adam
    for data in dataloader:
        img, labels = data

        # change the images to be 1D
        img = img.view(img.size(0), -1)

        # run images through ß-VAE
        out, mu, log_var = beta_vae(img)

        # run one optimization step on the loss function
        # the 'beta' term is the only difference between the VAE and ß-VAE loss functions
        beta_vae_loss = F.binary_cross_entropy(out, img) + (beta * KL(mu, log_var))
        beta_vae_optimizer.zero_grad()
        beta_vae_loss.backward()
        beta_vae_optimizer.step()

    # save images periodically
    if epoch % 10 == 0:
        pic = out.data * 0.5 + 0.5
        pic = pic.view(out.size(0), 1, 28, 28)
        save_image(pic, './img/' + str(epoch) + '_epochs.png')

    # plot loss
    update_viz(epoch, beta_vae_loss.item())
