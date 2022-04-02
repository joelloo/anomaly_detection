import os
import argparse
import torch
import torch.nn as nn
import wandb

from tqdm import tqdm

from datasets.testsets import TestDataSet
from models.vae import VariationalAutoencoderResNet

from torchvision.utils import save_image

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
parser.add_argument("--model_type", type=str, help="Autoencoder or VAE",
    default="vae")

args = parser.parse_args()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

test_data = TestDataSet(data_dir=args.dataset_dir)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=True, drop_last=False)

# Weights and biases logging
if args.wandb_entity:
    wandb.init(project="anomaly_detection", entity=args.wandb_entity)

wandb.config = {
    "epochs": 10
}

# Initialize model
model_types = ("vae", "ae")
assert args.model_type in model_types, f"Model type should be {model_types}"
if args.model_type == "vae":
    model = VariationalAutoencoderResNet(device, flows=None, latent_size=256, img_height=112, net_type='resnet18')
    wandb.config["threshold"] = 0.03
else:
    # Instantiate a ResNet-based autoencoder
    from pl_bolts.models.autoencoders import AE
    model = AE(input_height=112, enc_type='resnet18').to(device)
    wandb.config["threshold"] = 0.01

# Reconstruction loss for thresholding
criterion = nn.MSELoss(reduction="none")

for epoch in range(wandb.config["epochs"]):
    # Testing the performance of each model epoch
    model_path = os.path.join(args.model_dir, f"{args.model_type}_v1_e{epoch}.ckpt")
    model.load_state_dict(torch.load(model_path))

    ood_losses = []
    ind_losses = []

    for batch, data in tqdm(enumerate(test_loader), total=len(test_loader)):
        img, label = data
        img = img.to(device)

        if args.model_type == "vae":
            recon_img, _, _, _ = model(img)
        else:
            recon_img = model(img)

        loss = criterion(img, recon_img)
        mean_loss = loss.mean(dim=1)
        map_tensor = (mean_loss > 0.05).type(torch.FloatTensor)

        save_image(map_tensor, f'masks/{args.model_type}-e{epoch}-{batch}.png')

        # Concatenate losses for averaging later
        if label[0] == "ood":
            ood_losses.append(loss.mean().item())
        else:
            ind_losses.append(loss.mean().item())

    if args.wandb_entity:
        wandb.log({
            "In Distribution Loss": sum(ood_losses)/len(ood_losses),
            "Out of Distribution Loss": sum(ind_losses)/len(ind_losses)
        }, step=epoch)

