from torch.optim.lr_scheduler import ReduceLROnPlateau
from Dataloader import MNISTDataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from VAE import VAE
import numpy as np
import util
from torch.nn import functional as F
import matplotlib.pyplot as plt

class VAE_Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred, y):
        mu, var, out = pred
        r_loss = torch.sqrt(self.mse(out, y))
        kl_loss = -0.5 * (torch.sum(1 + var - torch.square(mu) - torch.exp(var)))

        return r_loss + kl_loss


def Train():
    mnist_dataset = MNISTDataset('../Dataset/MNIST/mnist_train.csv')

    batch_size = 32
    epochs = 10
    dataloader = DataLoader(mnist_dataset, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    vae = VAE()
    vae.to(device)

    optimizer = torch.optim.Adam(vae.parameters(), lr=0.001)
    criteria = VAE_Loss()

    best_loss = np.finfo(np.float64).max
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.9, patience=2)

    losses = []
    for epoch in range(epochs):
        for image, _ in dataloader:
            image = image.to(device)
            out = vae(image)

            loss = criteria(out, image)
            losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_loss = np.array(losses).mean()
        print("{} Epoch - avg. Loss : {:.4f}".format(epoch+1, avg_loss))

        if best_loss > avg_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epochs,
                'model_state_dict': vae.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, 'vae_best_model.pth')

            print("Saved Best Model avg. Loss : {:.4f}".format(avg_loss))

        scheduler.step(avg_loss)

    img, _ = mnist_dataset.__getitem__(0)
    img = torch.unsqueeze(img, 0)
    img = img.to(device)
    vae.Draw(img)

def DrawScatter():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    vae = VAE()
    vae.to(device)

    checkpoint = torch.load("vae_best_model.pth")
    vae.load_state_dict(checkpoint['model_state_dict'])
    vae.eval()

    mnist_dataset = MNISTDataset('../Dataset/MNIST/mnist_train.csv')

    util.DrawScatter(vae, mnist_dataset, device, True)

def Reconstruct():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    vae = VAE()
    vae.to(device)

    checkpoint = torch.load("vae_best_model.pth")
    vae.load_state_dict(checkpoint['model_state_dict'])
    vae.eval()

    mnist_dataset = MNISTDataset('../Dataset/MNIST/mnist_train.csv')

    util.z_latentReconstruct(vae, mnist_dataset, device)

if __name__ == '__main__':
    #Train()
    #DrawScatter()
    Reconstruct()
