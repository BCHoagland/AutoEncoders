import numpy as np
from visdom import Visdom

viz = Visdom()

title = 'VAE and ß-VAE Loss by Epoch'
win = None

def update_viz(epoch, vae_loss, beta_vae_loss):
    global win, title

    if win is None:
        title = title

        win = viz.line(
            X=np.array([[epoch, epoch]]),
            Y=np.array([[vae_loss, beta_vae_loss]]),
            win=title,
            opts=dict(
                title=title,
                fillarea=True,
                legend=['VAE', 'ß-VAE']
            )
        )
    else:
        viz.line(
            X=np.array([[epoch, epoch]]),
            Y=np.array([[vae_loss, beta_vae_loss]]),
            win=win,
            update='append'
        )

colors = np.array([[255, 0, 0],
                   [255, 100, 0],
                   [255, 200, 0],
                   [100, 255, 0],
                   [0, 255, 0],
                   [0, 150, 0],
                   [0, 150, 255],
                   [0, 0, 255],
                   [100, 0, 255],
                   [255, 0, 255]])

def beta_scatter(raw_coords, vae_coords, beta_vae_coords, labels):

    viz.scatter(
        X=raw_coords,
        Y=labels+1,
        win='latent raw',
        opts=dict(
            title='Raw Observations',
            markersize=4,
            markercolor=colors
        )
    )

    viz.scatter(
        X=vae_coords,
        Y=labels+1,
        win='latent VAE',
        opts=dict(
            title='Latent VAE Observations',
            markersize=4,
            markercolor=colors
        )
    )

    viz.scatter(
        X=beta_vae_coords,
        Y=labels+1,
        win='latent ß-VAE',
        opts=dict(
            title='Latent ß-VAE Observations',
            markersize=4,
            markercolor=colors
        )
    )

def bar(log_var):
    var = np.exp(log_var)
    viz.bar(
        X=var,
        win='Avg variance',
        opts=dict(
            title='Average Variance of Latent Variable Values',
            legend=['VAE', 'ß-VAE']
        )
    )
