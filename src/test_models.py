import os
import argparse
import torch
import torch.nn as nn

from tqdm import tqdm

from datasets.testsets import TestDataSet
from models.vae import VariationalAutoencoderResNet

from pdb import set_trace as bp

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_dir", type=str, help="Path to dataset",
    default="data/test")
parser.add_argument("--dataset_type", type=str, help="Pick which dataset to use: left, middle, right, all",
    default="all")
parser.add_argument("--model_dir", type=str, help="Path to save models",
    default="trained_models")
parser.add_argument("--wandb_entity", "-e", type=str, help="Weights and Biases entity",
    default=None)

args = parser.parse_args()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

test_data = TestDataSet(data_dir=args.dataset_dir)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=True, drop_last=False)

# Initialize model
model = VariationalAutoencoderResNet(device, flows=None, latent_size=256, img_height=112, net_type='resnet18')
model_path = os.path.join(args.model_dir, "vae_v1_e19.ckpt")
model.load_state_dict(torch.load(model_path))

# Reconstruction loss for thresholding
recon_loss = nn.MSELoss(reduction="mean")
ood_losses = []
ind_losses = []

for batch, data in tqdm(enumerate(test_loader), total=len(test_loader)):
    img, labels = data
    img = img.to(device)
    recon_img, _, _, _ = model(img)

    loss = recon_loss(img, recon_img)

    if labels[0] == 'ood':
        ood_losses.append(loss.item())
    else:
        ind_losses.append(loss.item())

bp()
