import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

import os
import argparse
import numpy as np
from tqdm import tqdm

from models.vae import VariationalAutoencoder, VariationalAutoencoderResNet
from datasets.datasets import GazeboSimDataset, load_all_datasets


parser = argparse.ArgumentParser()
parser.add_argument("--dataset_dir", type=str, help="Path to dataset",
    default="/home/joel/normflow_data")
parser.add_argument("--dataset_type", type=str, help="Pick which dataset to use: left, middle, right, all",
    default="all")
parser.add_argument("--model_dir", type=str, help="Path to save models",
    default="/home/joel/saved_models")

args = parser.parse_args()

### Train convolutional variational autoencoder ###

# Set up model hyperparameters
n_bottleneck = 256
n_flows = 16
n_epochs = 50
batch_size = 64
train_partition_fraction = 1.0
identifier_str = 'vae_v1'

enable_cuda = True
device = torch.device('cuda' if torch.cuda.is_available() and enable_cuda else 'cpu')

# Instantiate a variational autoencoder
# vae = VariationalAutoencoder(device).to(device)
vae = VariationalAutoencoderResNet(device, flows=None, latent_size=256, img_height=112, net_type='resnet18')

# Load dataset
datasets_map = load_all_datasets(args.dataset_dir)

if args.dataset_type is "all":
    datasets_list = list(datasets_map.values())
    train_dataset = torch.utils.data.ConcatDataset(datasets_list)
else:
    train_dataset = datasets_map[args.dataset_type]

train_partition_len = int(np.floor(train_partition_fraction * len(train_dataset)))
val_partition_len = len(train_dataset) - train_partition_len
train_set, val_set = torch.utils.data.random_split(train_dataset, [train_partition_len, val_partition_len])

train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=False)

print("Loaded data: ", args.dataset_type)
print("Train set: ", len(train_set))
print("Val set: ", len(val_set))

# Train the VAE
optimizer =  torch.optim.Adam(vae.parameters(), lr=1e-4, weight_decay=1e-4)
recon_loss = nn.MSELoss(reduction='sum')

for epoch in range(n_epochs):
    progressbar = tqdm(enumerate(train_loader), total=len(train_loader))
    for batch_n, x in progressbar:
        x = x.to(device)
        optimizer.zero_grad()
        outputs, mu, sigma, var_loss = vae(x)
        loss = recon_loss(outputs, x) + var_loss
        loss.backward()
        optimizer.step()
        progressbar.update()
    progressbar.close()
    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_n * len(x), len(train_loader.dataset),
                       100. * batch_n / len(train_loader),
                       loss.item()))
    if epoch % 10 == 9:
        print(f'Saving checkpoint for epoch {epoch}...')
        torch.save(vae.state_dict(), os.path.join(args.model_dir, f'{identifier_str}_e{epoch}.ckpt'))

# Save the final checkpoint
torch.save(vae.state_dict(), os.path.join(f'{identifier_str}_e{epoch}.ckpt'))
