from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt
import numpy as np

def DrawScatter(model, Dataset, device, show=False):
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
    if show:
        plt.show()

def z_latentReconstruct(model, Dataset, device):
    DrawScatter(model, Dataset, device)
    x = 0
    y = np.linspace(-0.04, 0.04, 25)

    xv, yv = np.meshgrid(x, y)
    xv = xv.flatten()
    yv = yv.flatten()

    z_grid = np.array(list(zip(xv, yv)))

    plt.scatter(z_grid[:, 0], z_grid[:, 1], c='black'  # , cmap='rainbow' , c= example_labels
                , alpha=1, s=5)

    plt.show()

    data = torch.from_numpy(z_grid).type(torch.float).to(device)
    reconst = model.Reconstruct(data).detach().cpu().numpy()

    fig = plt.figure(figsize=(5, 5))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    for i in range(5 ** 2):
        ax = fig.add_subplot(5, 5, i + 1)
        ax.axis('off')
        ax.imshow(reconst[i, 0, :, :], cmap='Greys')
    plt.show()