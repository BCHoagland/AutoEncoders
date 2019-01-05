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

def scatter(coords, labels):
    viz.scatter(
        X=coords,
        Y=labels + 1,
        win='VAE latent space',
        opts=dict(
            # legend=(labels - 1).tolist(),
            title='Distribution of Latent Variables in VAE',
            markercolor=colors,
            markersize=4
        )
    )

def beta_scatter(coords, labels):
    viz.scatter(
        X=coords,
        Y=labels + 1,
        win='ß-VAE latent space',
        opts=dict(
            # legend=(labels - 1).tolist(),
            title='Distribution of Latent Variables in ß-VAE',
            markercolor=colors,
            markersize=4
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
