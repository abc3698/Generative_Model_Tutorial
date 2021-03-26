from Dataloader import MNISTDataset
from AutoEncoder import AutoEncoder
from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt

def DrawScatter(model, Dataset, device):
    batch_size = 5000
    dataloader = DataLoader(Dataset, batch_size=batch_size, shuffle=True)

    dataloader_iterator = iter(dataloader)
    imgs, labels = next(dataloader_iterator)
    imgs = imgs.to(device)

    z_points = model.latent_vector(imgs).detach().cpu().numpy()

    plt.figure(figsize=(12, 12))
    plt.scatter(z_points[:, 0], z_points[:, 1], cmap='rainbow', c=labels
                , alpha=0.5, s=2)
    plt.colorbar()
    plt.show()