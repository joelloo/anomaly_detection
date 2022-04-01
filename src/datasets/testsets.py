import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Compose, CenterCrop, Resize, ToTensor

from pdb import set_trace as bp

class TestDataSet(Dataset):
    def __init__(self, data_dir="data/test", transforms=None):
        """ 
        Expected structure of data_dir:
        data_dir
            - ind
                - left1
                - left2
                - middle
                - right1
                - right2
            - ood
        """
        self.data_dir = data_dir
        self.transforms = transforms if transforms else Compose([ToTensor(), Resize(128), CenterCrop(112)])
        
        # Identify all data
        self.labels = os.listdir(data_dir)
        self.entries = []
        for label in self.labels:
            start_path = os.path.join(data_dir, label)
            for dirpath, subdirs, files in os.walk(start_path):
                for filename in files:
                    full_path = os.path.join(dirpath, filename)
                    if os.path.isfile(full_path):
                        self.entries.append((full_path, label))

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        img_path, label = self.entries[idx]
        image = Image.open(img_path)
        image = self.transforms(image)
        return image, label

    