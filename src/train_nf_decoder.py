import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, models, transforms

import os
import argparse
import numpy as np
from tqdm import tqdm

from datasets.datasets import GazeboSimDataset, load_all_datasets

from pl_bolts.models.autoencoders.components import resnet18_decoder

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_dir", type=str, help="Path to dataset",
    default="/home/joel/normflow_data")
parser.add_argument("--dataset_type", type=str, help="Pick which dataset to use: left, middle, right, all",
    default="all")
parser.add_argument("--model_dir", type=str, help="Path to save models",
    default="/home/joel/source/saved_models")

args = parser.parse_args()

### Train RealNVP normalising flow with frozen ResNet encoder ###
n_bottleneck = 1000
n_epochs = 50
batch_size = 64
train_partition_fraction = 1.0
identifier_str = 'nf'

enable_cuda = True
device = torch.device('cuda' if torch.cuda.is_available() and enable_cuda else 'cpu')
print(f'Device: {device}')

# Initialise and freeze the ResNet18 encoder
encoder = models.resnet18(pretrained=True).to(device)

# Instantiate the ResNet18 decoder to train
decoder = resnet18_decoder(n_bottleneck, 112, first_conv=False, maxpool1=False).to(device)

# Load the dataset
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

print("Loaded data.")
print("Train set: ", len(train_set))
print("Val set: ", len(val_set))

# Train the AE
optimizer =  torch.optim.Adam(decoder.parameters(), lr=1e-4, weight_decay=1e-4)
loss_fn = nn.MSELoss()

for epoch in range(n_epochs):
    print(f'Learning rate: {optimizer.param_groups[0]["lr"]}')
    progressbar = tqdm(enumerate(train_loader), total=len(train_loader))
    for batch_n, x in progressbar:
        x = x.to(device)
        optimizer.zero_grad()

        z = encoder(x)
        y = decoder(z)
        loss = loss_fn(y, x)

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
        torch.save(decoder.state_dict(), os.path.join(args.model_dir, f'{identifier_str}_e{epoch}.ckpt'))

# Save the final checkpoint
torch.save(decoder.state_dict(), os.path.join(args.model_dir, f'{identifier_str}_e{epoch}.ckpt'))