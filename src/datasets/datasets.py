import os
import glob
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import Compose, CenterCrop, Resize
from torchvision.io import read_image

# import matplotlib.pyplot as plt

class GazeboSimDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.img_files = glob.glob(img_dir + "/*.jpeg")
        self.transform = None
        if transform is None:
            self.transform = Compose([Resize(256), CenterCrop(224)])

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        image = read_image(img_path)
        if self.transform:
            image = self.transform(image)
        return image

# if __name__ == "__main__":
#     print("Loading data")
#     train_data = GazeboSimDataset('/home/joel/Downloads/images-arm/data/middle')
#     x0 = train_data[56].numpy().swapaxes(0, 2).swapaxes(0, 1)
#     plt.imshow(x0)
#     plt.show()