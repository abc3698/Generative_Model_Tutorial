from torch.optim.lr_scheduler import ReduceLROnPlateau
from Dataloader import MNISTDataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from AutoEncoder import AutoEncoder
import numpy as np
import util
import matplotlib.pyplot as plt

class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred, y):
        return torch.sqrt(self.mse(pred, y))

def Train():
    mnist_dataset = MNISTDataset('../Dataset/MNIST/mnist_train.csv')

    batch_size = 32
    epochs = 10
    dataloader = DataLoader(mnist_dataset, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    autoEncoder = AutoEncoder()
    autoEncoder.to(device)

    optimizer = torch.optim.Adam(autoEncoder.parameters(), lr=0.001)
    criteria = RMSELoss()

    best_loss = np.finfo(np.float64).max
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.9, patience=2)

    losses = []
    for epoch in range(epochs):
        for image, _ in dataloader:
            image = image.to(device)
            out = autoEncoder(image)

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
                'model_state_dict': autoEncoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, 'best_model.pth')

            print("Saved Best Model avg. Loss : {:.4f}".format(avg_loss))

        scheduler.step(avg_loss)

    img, _ = mnist_dataset.__getitem__(0)
    img = torch.unsqueeze(img, 0)
    img = img.to(device)
    autoEncoder.Draw(img)

def DrawScatter():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    autoEncoder = AutoEncoder()
    autoEncoder.to(device)

    checkpoint = torch.load("best_model.pth")
    autoEncoder.load_state_dict(checkpoint['model_state_dict'])
    autoEncoder.eval()

    mnist_dataset = MNISTDataset('../Dataset/MNIST/mnist_train.csv')

    util.DrawScatter(autoEncoder, mnist_dataset, device, True)

def Reconstruct():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    autoEncoder = AutoEncoder()
    autoEncoder.to(device)

    checkpoint = torch.load("best_model.pth")
    autoEncoder.load_state_dict(checkpoint['model_state_dict'])
    autoEncoder.eval()

    mnist_dataset = MNISTDataset('../Dataset/MNIST/mnist_train.csv')

    util.z_latentReconstruct(autoEncoder, mnist_dataset, device)

if __name__ == '__main__':
    #Train()
    #DrawScatter()
    Reconstruct()
