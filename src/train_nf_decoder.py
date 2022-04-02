import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, models, transforms

import os
import argparse
import numpy as np
from tqdm import tqdm

import wandb

from datasets.datasets import  load_all_datasets

from pl_bolts.models.autoencoders.components import resnet18_decoder

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_dir", type=str, help="Path to dataset",
    default="data/train")
parser.add_argument("--dataset_type", type=str, help="Pick which dataset to use: left, middle, right, all",
    default="all")
parser.add_argument("--model_dir", type=str, help="Path to save models",
    default="trained_models")
parser.add_argument("--wandb_entity", "-e", type=str, help="Weights and Biases entity",
    default=None)

args = parser.parse_args()

### Train RealNVP normalising flow with frozen ResNet encoder ###
train_partition_fraction = 1.0
identifier_str = 'nf'

# Load dataset
workdir = os.getcwd()
full_model_dir = os.path.join(workdir, args.model_dir)
datasets_map = load_all_datasets(os.path.join(workdir, args.dataset_dir))

# Weights and biases logging
if args.wandb_entity:
    wandb.init(project="anomaly_detection", entity=args.wandb_entity)
    

wandb.config = {
    "learning_rate": 1e-4,
    "epochs": 10,
    "batch_size": 32,
    "weight_decay": 1e-4,
    "n_bottleneck" : 1000,
}

enable_cuda = True
device = torch.device('cuda:1' if torch.cuda.is_available() and enable_cuda else 'cpu')
print(f'Device: {device}')

# Initialise and freeze the ResNet18 encoder
encoder = models.resnet18(pretrained=True).to(device)

# Instantiate the ResNet18 decoder to train
decoder = resnet18_decoder(wandb.config["n_bottleneck"], 224, first_conv=False, maxpool1=False).to(device)

if args.dataset_type == "all":
    datasets_list = list(datasets_map.values())
    train_dataset = torch.utils.data.ConcatDataset(datasets_list)
else:
    train_dataset = datasets_map[args.dataset_type]

train_partition_len = int(np.floor(train_partition_fraction * len(train_dataset)))
val_partition_len = len(train_dataset) - train_partition_len
train_set, val_set = torch.utils.data.random_split(train_dataset, [train_partition_len, val_partition_len])

train_loader = torch.utils.data.DataLoader(train_set, batch_size=wandb.config["batch_size"], shuffle=True, drop_last=False)

print("Loaded data.")
print("Train set: ", len(train_set))
print("Val set: ", len(val_set))

# Train the AE
optimizer =  torch.optim.Adam(decoder.parameters(), lr=wandb.config["learning_rate"], weight_decay=wandb.config["weight_decay"])
loss_fn = nn.MSELoss()

for epoch in range(wandb.config["epochs"]):
    print(f'Learning rate: {optimizer.param_groups[0]["lr"]}')
    run_loss = 0.0

    progressbar = tqdm(enumerate(train_loader), total=len(train_loader))
    for batch_n, x in progressbar:
        x = x.to(device)
        optimizer.zero_grad()

        z = encoder(x)
        y = decoder(z)
        loss = loss_fn(y, x)
        run_loss += loss

        loss.backward()
        optimizer.step()

        progressbar.update()
    progressbar.close()
    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_n * len(x), len(train_loader.dataset),
                       100. * batch_n / len(train_loader),
                       loss.item()))

    print(f'Saving checkpoint for epoch {epoch}...')
    torch.save(decoder.state_dict(), os.path.join(full_model_dir, f'{identifier_str}_e{epoch}.ckpt'))

    if args.wandb_entity:
        wandb.log({
            "avg_recon_loss": run_loss / batch_n,
        })
        wandb.log_artifact(os.path.join(full_model_dir, f'{identifier_str}_e{epoch}.ckpt'), name=f'nf-decoder-e{epoch}', type='nf-decoder')

# Save the final checkpoint
torch.save(decoder.state_dict(), os.path.join(full_model_dir, f'{identifier_str}_e{epoch}.ckpt'))