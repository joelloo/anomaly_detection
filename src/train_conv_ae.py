import torch
import torch.nn as nn

import os
import argparse
import numpy as np
from tqdm import tqdm

import wandb

from datasets.datasets import load_all_datasets

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

### Train vanilla convolutional autoencoder ###

# Set up model hyperparameters
train_partition_fraction = 1.0
identifier_str = 'ae_v1'

enable_cuda = True
device = torch.device('cuda:0' if torch.cuda.is_available() and enable_cuda else 'cpu')

# Instantiate a ResNet-based autoencoder
from pl_bolts.models.autoencoders import AE
ae = AE(input_height=112, enc_type='resnet18').to(device)

# Load the dataset
workdir = os.getcwd()
full_model_dir = os.path.join(workdir, args.model_dir)
datasets_map = load_all_datasets(os.path.join(workdir, args.dataset_dir))

# Weights and biases logging
config = {
  "learning_rate": 1e-4,
  "epochs": 50,
  "batch_size": 32,
  "weight_decay": 1e-4
}

if args.wandb_entity:
    wandb.init(
        id="ae-training",
        project="anomaly_detection", 
        entity=args.wandb_entity, 
        name="Train AE",
        config=config,
        resume="allow"
    )

if args.dataset_type == "all":
    datasets_list = list(datasets_map.values())
    train_dataset = torch.utils.data.ConcatDataset(datasets_list)
else:
    train_dataset = datasets_map[args.dataset_type]

train_partition_len = int(np.floor(train_partition_fraction * len(train_dataset)))
val_partition_len = len(train_dataset) - train_partition_len
train_set, val_set = torch.utils.data.random_split(train_dataset, [train_partition_len, val_partition_len])

train_loader = torch.utils.data.DataLoader(train_set, batch_size=config["batch_size"], shuffle=True, drop_last=False)

print("Loaded data.")
print("Train set: ", len(train_set))
print("Val set: ", len(val_set))

# Train the AE
optimizer =  torch.optim.Adam(ae.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
loss_fn = nn.MSELoss()

for epoch in range(config["epochs"]):
    progressbar = tqdm(enumerate(train_loader), total=len(train_loader))
    run_loss = 0.0
    for batch_n, x in progressbar:
        x = x.to(device)
        optimizer.zero_grad()
        z = ae(x)
        loss = loss_fn(z, x)
        loss.backward()
        run_loss += loss.item()
        optimizer.step()
        progressbar.update()
    progressbar.close()
    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_n * len(x), len(train_loader.dataset),
                       100. * batch_n / len(train_loader),
                       loss.item()))

    if args.wandb_entity:
        wandb.log({"avg_recon_loss": run_loss / batch_n}, step=epoch)

    print(f'Saving checkpoint for epoch {epoch}...')
    torch.save(ae.state_dict(), os.path.join(full_model_dir, f'{identifier_str}_e{epoch}.ckpt'))
    if args.wandb_entity:
        wandb.log_artifact(os.path.join(full_model_dir, f'{identifier_str}_e{epoch}.ckpt'), name=f'ae-e{epoch}', type='ae-models') 

# Save the final checkpoint
torch.save(ae.state_dict(), os.path.join(full_model_dir, f'{identifier_str}_e{epoch}.ckpt'))
if args.wandb_entity:
        wandb.log_artifact(os.path.join(full_model_dir, f'{identifier_str}_e{epoch}.ckpt'), name=f'ae-e{epoch}', type='ae-models') 
