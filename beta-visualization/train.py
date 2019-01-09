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

# visloader = DataLoader(dataset, batch_size=len(dataset))
visloader = DataLoader(dataset, batch_size=1000)

# create vae and its optimizer
vae = VAE()
vae_optimizer = optim.Adam(vae.parameters(), lr=lr)

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

        # normal VAE
        out1, mu1, log_var1 = vae(img)
        vae_loss = F.binary_cross_entropy(out1, img) + KL(mu1, log_var1)
        vae_optimizer.zero_grad()
        vae_loss.backward()
        vae_optimizer.step()

        # ß-VAE
        # the 'beta' term is the only difference between VAE and ß-VAE
        out2, mu2, log_var2 = beta_vae(img)
        beta_vae_loss = F.binary_cross_entropy(out2, img) + (beta * KL(mu2, log_var2))
        beta_vae_optimizer.zero_grad()
        beta_vae_loss.backward()
        beta_vae_optimizer.step()

    # save images periodically
    if epoch % 10 == 0:
        pic = out1.data.view(out1.size(0), 1, 28, 28)
        save_image(pic, './img/vae_' + str(epoch) + '_epochs.png')

        pic = out2.data.view(out2.size(0), 1, 28, 28)
        save_image(pic, './img/ß_vae_' + str(epoch) + '_epochs.png')

    with torch.no_grad():
        for img, labels in visloader:
            img = img.view(img.size(0), -1)

            scatter(vae.encode(img).numpy()[:,:3], labels.numpy())
            beta_scatter(beta_vae.encode(img).numpy()[:,:3], labels.numpy())

            _, _, log_var1 = vae(img)
            _, _, log_var2 = beta_vae(img)
            log_vars = torch.t(torch.stack((torch.mean(log_var1, dim=0), torch.mean(log_var2, dim=0))))
            bar(log_vars.numpy())

            break

    # plot loss
    update_viz(epoch, vae_loss.item(), beta_vae_loss.item())
