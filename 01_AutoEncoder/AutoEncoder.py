import torch.nn as nn
from Encoder import Encoder
from Decoder import Decoder
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

class AutoEncoder(nn.Module):
    def __init__(self, latent_dim=2):
        super().__init__()

        self.Encoder = Encoder(latent_dim)
        self.Decoder = Decoder(latent_dim)
        self.inv_trans = transforms.Compose([
            transforms.Normalize(mean=(-0.5/0.5), std=(1./0.5))
            ])

    def forward(self, inputs):
        x = self.Encoder(inputs)
        x = self.Decoder(x)
        return x

    def Draw(self, input):
        out = self.forward(input)
        out = self.inv_trans(out).detach().cpu().numpy()
        out = np.squeeze(out, 0)
        out = np.squeeze(out, 0)

        plt.imshow(out, interpolation='none', cmap='Blues')
        plt.show()

