from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import pandas
import os
from torchvision import transforms
os.environ['KMP_DUPLICATE_LIB_OK']='True'

class MNISTDataset(Dataset):

    def __init__(self, csv_file):
        self.data_df = pandas.read_csv(csv_file, header=None)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5), std=(0.5))
            ])

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, index):
        label = self.data_df.iloc[index, 0]
        image_values = self.data_df.iloc[index, 1:].values.reshape(28, 28) / 255.
        image_values = self.transform(image_values).float()
        return image_values, label

    def plot_image(self, index):
        img = self.data_df.iloc[index, 1:].values.reshape(28, 28)
        plt.title("label = " + str(self.data_df.iloc[index, 0]))
        plt.imshow(img, interpolation='none', cmap='Blues')
        plt.show()