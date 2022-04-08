import torch
import torch.nn as nn
from torchvision import datasets, models
from pl_bolts.models.autoencoders import resnet18_decoder, AE

import cv2
import numpy as np
import argparse
import matplotlib.pyplot as plt
from datasets.datasets import load_all_datasets, GazeboSimDataset
from datasets.testsets import TestDataSet
from models.realnvp import LatentMaskedAffineCoupling, MLP, NormalisingFlow

from tqdm import tqdm
import json
import sys
import glob
import os
import re

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", type=str, help="Path to models",
    default="/home/joel/Downloads/nf_models")
parser.add_argument("--encoder_dir", type=str, help="Path to encoder weights",
    default="/home/joel/Downloads/trained_models/ae_v1_e49.ckpt")

args = parser.parse_args()

enable_cuda = True
device = torch.device('cuda:0' if torch.cuda.is_available() and enable_cuda else 'cpu')
print(f'Device: {device}')

# Load models
state = torch.load(args.encoder_dir, map_location=device)
ae = AE(input_height=112, enc_type='resnet18')
ae.load_state_dict(state)
ae.to(device)
ae.eval()

# Helper functions
def generate_rmse_loss_heatmap(im_in, im_out):
    sqdiff = (im_in - im_out)**2
    rmse = np.sqrt(np.mean(sqdiff, axis=2))
    rmse_cmap = plt.cm.jet(rmse)[:, :, :-1]
    combined = cv2.addWeighted(rmse_cmap, 0.4, im_in.astype(np.float64), 0.6, 0.0)
    return rmse, combined

def visualise(axs, test_ims, recon_ims):
    batch_size = test_ims.shape[0]
    for i in range(batch_size):
        axs[i, 0].imshow(test_ims[i])
        axs[i, 1].imshow(recon_ims[i])
        _, combined = generate_rmse_loss_heatmap(test_ims[i], recon_ims[i])
        axs[i, 2].imshow(combined)

# Set up the normalising flow
model = "e79_256_64.ckpt"
tokens = re.split('[_.]', model)
print(tokens)
n_bottleneck = int(tokens[1])
n_flows = int(tokens[2])
print(f'Setting up model with {n_bottleneck} bottleneck and {n_flows} flows')

b = torch.tensor(n_bottleneck // 2 * [0, 1] + n_bottleneck % 2 * [0])
flows = []
for i in range(n_flows):
    st_net = MLP(n_bottleneck, 1024, 3)
    if i % 2 == 0:
        flows += [LatentMaskedAffineCoupling(b, st_net)]
    else:
        flows += [LatentMaskedAffineCoupling(1 - b, st_net)]

prior = torch.distributions.normal.Normal(loc=0.0, scale=1.0)

nf = NormalisingFlow(flows, prior, device, first_linear=None)
nf.load_state_dict(torch.load(os.path.join(args.model_dir, model)))
nf.to(device)
nf.eval()


# Visualise samples from the normalising flow
samples = nf.sample(sample_shape=[4, 256])
recon = ae.decoder(samples)
x_np = recon.view((4, 3, 112, 112)).cpu().detach().numpy()

fig, axs = plt.subplots(2, 2)
axs[0, 0].imshow(x_np[0].swapaxes(0, 2).swapaxes(0, 1))
axs[0, 1].imshow(x_np[1].swapaxes(0, 2).swapaxes(0, 1))
axs[1, 0].imshow(x_np[2].swapaxes(0, 2).swapaxes(0, 1))
axs[1, 1].imshow(x_np[3].swapaxes(0, 2).swapaxes(0, 1))

plt.suptitle(f'RealNVP ({n_flows} flows)')
plt.show()
