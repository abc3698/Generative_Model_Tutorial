from torch.optim.lr_scheduler import ReduceLROnPlateau
from Dataloader import MNISTDataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from CVAE import CVAE
import numpy as np
import util

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

    cvae = CVAE()
    cvae.to(device)

    optimizer = torch.optim.Adam(cvae.parameters(), lr=0.001)
    criteria = VAE_Loss()

    best_loss = np.finfo(np.float64).max
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.9, patience=2)

    losses = []
    for epoch in range(epochs):
        for image, label in dataloader:
            image = image.to(device)

            out = cvae(image, label)

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
                'model_state_dict': cvae.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, 'cvae_best_model.pth')

            print("Saved Best Model avg. Loss : {:.4f}".format(avg_loss))

        scheduler.step(avg_loss)

    img, labels = mnist_dataset.__getitem__(0)
    img = torch.unsqueeze(img, 0).to(device)
    labels = np.expand_dims(labels, 0)
    cvae.Draw(img, labels)

def DrawScatter():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    cvae = CVAE()
    cvae.to(device)

    checkpoint = torch.load("cvae_best_model.pth")
    cvae.load_state_dict(checkpoint['model_state_dict'])
    cvae.eval()

    mnist_dataset = MNISTDataset('../Dataset/MNIST/mnist_train.csv')

    util.DrawScatter(cvae, mnist_dataset, device, True)

def Reconstruct():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    cvae = CVAE()
    cvae.to(device)

    checkpoint = torch.load("cvae_best_model.pth")
    cvae.load_state_dict(checkpoint['model_state_dict'])
    cvae.eval()

    mnist_dataset = MNISTDataset('../Dataset/MNIST/mnist_train.csv')

    util.z_latentReconstruct(cvae, mnist_dataset, device)

if __name__ == '__main__':
    #Train()
    #DrawScatter()
    Reconstruct()
