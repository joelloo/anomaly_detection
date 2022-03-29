import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

import os
import argparse
import numpy as np
from tqdm import tqdm

from datasets.datasets import GazeboSimDataset, load_all_datasets


parser = argparse.ArgumentParser()
parser.add_argument("--dataset_dir", type=str, help="Path to dataset",
    default="/data/joel/anomaly_detection/normflow_data")
parser.add_argument("--dataset_type", type=str, help="Pick which dataset to use: left, middle, right, all",
    default="all")
parser.add_argument("--model_dir", type=str, help="Path to save models",
    default="/data/joel/anomaly_detection/saved_models")

args = parser.parse_args()

### Train RealNVP normalising flow with frozen ResNet encoder ###