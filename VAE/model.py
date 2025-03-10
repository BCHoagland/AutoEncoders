import torch
import torch.nn as nn

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.ELU()
        )

        self.mu = nn.Sequential(
            nn.Linear(64, 10)
        )

        self.log_var = nn.Sequential(
            nn.Linear(64, 10)
        )

        self.decoder = nn.Sequential(
            nn.Linear(10, 64),
            nn.ELU(),
            nn.Linear(64, 128),
            nn.ELU(),
            nn.Linear(128, 28 * 28),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)

        mu = self.mu(x)
        log_var = self.log_var(x)

        # z = mu + (std_dev * eps), where eps ~ N(0,1)
        z = mu + torch.mul(torch.exp(log_var / 2), torch.randn_like(log_var))

        x_hat = self.decoder(z)

        return x_hat, mu, log_var

    def encode(self, x):
        x = self.encoder(x)

        mu = self.mu(x)
        log_var = self.log_var(x)
        z = mu + torch.mul(torch.exp(log_var / 2), torch.randn_like(log_var))
        return z

    def decode(self, z):
        return self.decoder(z)
