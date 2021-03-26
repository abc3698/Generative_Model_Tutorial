import torch.nn as nn

class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)

class Decoder(nn.Module):
    def __init__(self, latent_dim=2):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 3136),
            Reshape(-1, 64, 7, 7),
            self.MakeBlocK(64, 64, 1),
            self.MakeBlocK(64, 64, 2),
            self.MakeBlocK(64, 32, 2),
            self.MakeBlocK(32, 1, 1),
        )

    def MakeBlocK(self, in_ch, out_ch, stride, last_layer = False):
        if(stride == 1):
            conv = nn.ConvTranspose2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, stride=stride, padding=1, output_padding=0)
        else:
            conv = nn.ConvTranspose2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, stride=stride, padding=1, output_padding=1)

        if(last_layer):
            act = nn.LeakyReLU()
        else:
            act = nn.Tanh()
        return nn.Sequential(
            conv, act
        )

    def forward(self, inputs):
        return self.model(inputs)