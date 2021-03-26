import torch.nn as nn
from Encoder import Encoder
from Decoder import Decoder
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

class VAE(nn.Module):
    def __init__(self, latent_dim=2):
        super().__init__()

        self.Encoder = Encoder(latent_dim)
        self.Decoder = Decoder(latent_dim)
        self.inv_trans = transforms.Compose([
            transforms.Normalize(mean=(-0.5/0.5), std=(1./0.5))
            ])

    def forward(self, inputs):
        x = self.Encoder(inputs)
        recon = self.Decoder(x)
        return [x[:,0], x[:,1], recon]

    def Draw(self, input):
        _, _, out = self.forward(input)
        out = self.inv_trans(out).detach().cpu().numpy()
        out = np.squeeze(out, 0)
        out = np.squeeze(out, 0)

        plt.imshow(out, interpolation='none', cmap='Blues')
        plt.show()

    def latent_vector(self, inputs):
        x = self.Encoder(inputs)
        return x

    def Reconstruct(self, inputs):
        x = self.Decoder(inputs)
        return x


