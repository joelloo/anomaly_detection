import cv2
import os
import torch
import torch.nn as nn
import wandb
import numpy as np

from tqdm import tqdm

from datasets.testsets import TestDataSet
from models.vae import VariationalAutoencoderResNet
from datasets.datasets import load_all_datasets

from torchvision.transforms import Compose, Normalize
from utils import construct_parser, find_threshold

import matplotlib.pyplot as plt

from pdb import set_trace as bp

parser = construct_parser()
parser.add_argument("--train_data", type=str, help="Directory for train data",
        default="data/train")
parser.add_argument("--model_type", type=str, help="model_type",
        default="vae")
args = parser.parse_args()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

test_data = TestDataSet(data_dir=args.dataset_dir)
workdir = os.getcwd()
full_model_dir = os.path.join(workdir, args.model_dir)
datasets_map = load_all_datasets(os.path.join(workdir, args.train_data))
test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=True, drop_last=False)

if args.dataset_type == "all":
    datasets_list = list(datasets_map.values())
    train_dataset = torch.utils.data.ConcatDataset(datasets_list)
else:
    train_dataset = datasets_map[args.dataset_type]

config = {
    "epochs": 50
}

# Weights and biases logging
if args.wandb_entity:
    wandb.init(project="anomaly_detection", entity=args.wandb_entity, config=config)

# Initialize model
model_types = ("vae", "ae")
assert args.model_type in model_types, f"Model type should be {model_types}"
if args.model_type == "vae":
    model = VariationalAutoencoderResNet(device, flows=None, latent_size=256, img_height=224, net_type='resnet18')
else:
    # Instantiate a ResNet-based autoencoder
    from pl_bolts.models.autoencoders import AE
    model = AE(input_height=224, enc_type='resnet18').to(device)

# Reconstruction loss for thresholding
criterion = nn.MSELoss(reduction="none")

UnNormalize = Compose([
    Normalize(mean = [ 0., 0., 0. ], std = [ 1/0.229, 1/0.224, 1/0.225 ]),
    Normalize(mean = [ -0.485, -0.456, -0.406 ], std = [ 1., 1., 1. ]),
])

for epoch in range(config["epochs"]):
    # Testing the performance of each model epoch
    model_path = os.path.join(args.model_dir, f"{args.model_type}_v1_e{epoch}.ckpt")
    model.load_state_dict(torch.load(model_path))

    threshold_mean, threshold_std = find_threshold(model, train_dataset, device, args.model_type)

    ood_losses = []
    ind_losses = []
    predictions = []
    true_labels = []

    for batch, data in tqdm(enumerate(test_loader), total=len(test_loader)):
        img, label = data
        img = img.to(device)

        if args.model_type == "vae":
            recon_img, _, _, _ = model(img)
        else:
            recon_img = model(img)

        loss = criterion(img, recon_img)
        mean_loss = loss.mean(dim=1)
        map_tensor = (mean_loss > threshold_mean + threshold_std).type(torch.FloatTensor)

        # Concatenate losses for averaging later
        if label[0] == "ood":
            ood_losses.append(loss.mean().item())
        else:
            ind_losses.append(loss.mean().item())

        # For calculating accuracies
        predictions.append(loss.mean().item() > threshold_mean + threshold_std)
        true_labels.append(label[0] == "ood")

        # Save every 50th image as reconstruction visualization
        if batch % 50 == 0:
            recon_img = UnNormalize(recon_img)
            try:
                plt.imsave(f"images/recon/{args.model_type}/recon-e{epoch}-{batch}.png", recon_img.detach().squeeze(0).permute(1,2,0).cpu().numpy())
            except ValueError as e:
                print(e)

    predictions = np.asarray(predictions, dtype=int)
    true_labels = np.asarray(true_labels, dtype=int)

    acc = (predictions == true_labels).mean()
    print(acc)
    if args.wandb_entity:
        wandb.log({
            "In Distribution Loss": sum(ood_losses)/len(ood_losses),
            "Out of Distribution Loss": sum(ind_losses)/len(ind_losses),
            "Accuracy": acc
        }, step=epoch)

