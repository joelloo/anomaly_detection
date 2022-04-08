import os
import glob
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import Compose, CenterCrop, Resize, ToTensor, Normalize

from PIL import Image


class GazeboSimDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.img_files = glob.glob(img_dir + "/*.jpeg")
        self.transform = None
        if transform is None:
            normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            self.transform = Compose([ToTensor(), Resize(256), CenterCrop(224), normalize])
            # self.transform = Compose([ToTensor(), Resize(128), CenterCrop(112)])

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        return image


def load_all_datasets(path, transform=None):
    subdirs = glob.glob(os.path.join(path, "*/"))
    print("Loading datasets from ", subdirs)

    datasets = {}
    for subdir in subdirs:
        name = os.path.basename(os.path.normpath(subdir))
        datasets[name] = GazeboSimDataset(subdir, transform=transform)

    return datasets

# if __name__ == "__main__":
#     print("Loading data")
#     train_data = GazeboSimDataset('/home/joel/Downloads/images-arm/data/middle')
#     x0 = train_data[56].numpy().swapaxes(0, 2).swapaxes(0, 1)
#     plt.imshow(x0)
#     plt.show()