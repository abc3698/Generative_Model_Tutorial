import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, latent_dim=2):
        super().__init__()

        self.model = nn.Sequential(
            self.MakeBlocK(1, 32, 1),
            self.MakeBlocK(32, 64, 2),
            self.MakeBlocK(64, 64, 2),
            self.MakeBlocK(64, 64, 1),
            nn.Flatten(),
            nn.Linear(3136, latent_dim)
        )

    def MakeBlocK(self, in_ch, out_ch, stride):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, stride=stride, padding=1),
            nn.LeakyReLU()
        )

    def forward(self, inputs):
        return self.model(inputs)
