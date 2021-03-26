import torch.nn as nn
import torch

class Encoder(nn.Module):
    def __init__(self, latent_dim=2, device="cuda"):
        super().__init__()

        self.encode = nn.Sequential(
            self.MakeBlocK(2, 32, 1),
            self.MakeBlocK(32, 64, 2),
            self.MakeBlocK(64, 64, 2),
            self.MakeBlocK(64, 64, 1),
            nn.Flatten(),
        )

        self.mu = nn.Linear(3136, latent_dim)
        self.var = nn.Linear(3136, latent_dim)
        self.device = device

    def MakeBlocK(self, in_ch, out_ch, stride):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, stride=stride, padding=1),
            nn.LeakyReLU()
        )

    def forward(self, inputs, labels):
        batch, _, h, w = inputs.shape
        conds = torch.zeros((batch, 1, h, w))
        conds = [torch.full_like(conds[idx], label) for idx, label in enumerate(labels)]
        conds = torch.stack(conds).type(torch.float).to(self.device)

        inputs = torch.cat((inputs, conds), 1)
        x = self.encode(inputs)

        mu = self.mu(x)
        var = self.var(x)

        # Sampling
        eps = torch.randn_like(mu)
        return mu + torch.exp(var * 0.5) * eps
