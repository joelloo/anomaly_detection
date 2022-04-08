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


enable_cuda = True
device = torch.device('cuda:0' if torch.cuda.is_available() and enable_cuda else 'cpu')
print(f'Device: {device}')

# Load models
state = torch.load("/home/joel/Downloads/trained_models/ae_v1_e49.ckpt", map_location=device)
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

# Set up normalising flow
n_bottleneck = 256
n_flows = 32
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
nf.load_state_dict(torch.load("/home/joel/Downloads/nf_e69_size256_flows32.ckpt"))
nf.to(device)
nf.eval()

# Compute the losses for binary label in the test set
test_data = TestDataSet(data_dir="/home/joel/Downloads/test_data")
test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False, drop_last=False)

losses = dict()

for batch, data in tqdm(enumerate(test_loader), total=len(test_loader)):
    img, label = data
    img = img.to(device)
    label = label[0]

    z = ae.encoder(img)
    z = ae.fc(z)
    z, log_det = nf(z)
    log_prob_z = nf.get_prior_log_prob(z)
    loss = -log_prob_z.mean() - log_det.mean()

    if label in losses.keys():
        losses[label].append(loss.cpu().item())
    else:
        losses[label] = [loss.cpu().item()]


# Determine statistics of the losses
with open("losses.json", "w") as fp:
    json.dump(losses, fp)

stats = dict()
for key in losses.keys():
    losses_array = np.array(losses[key])
    key_mean = np.mean(losses_array)
    key_std = np.std(losses_array)
    stats[key] = (key_mean, key_std)
    print(key, ": ", key_mean, key_std)

# Set the classification threshold as 2 std deviations from mean
ood_mean, ood_std = stats['ood']
ood_loss_threshold = ood_mean + 1 * ood_std
print("Threshold:" , ood_loss_threshold)

results = {'ood': {'pass': 0, 'fail': 0}, 'ind': {'pass': 0, 'fail': 0}}
for label, loss_array in losses.items():
    for loss in loss_array:
        if loss < ood_loss_threshold:
            if label == 'ood':
                results['ood']['pass'] += 1
            else:
                results['ind']['fail'] += 1
        else:
            if label == 'ood':
                results['ood']['fail'] += 1
            else:
                results['ind']['pass'] += 1

print(results)


# Visualise samples from the normalising flow
samples = nf.sample(sample_shape=[4, 256])
recon = ae.decoder(samples)
x_np = recon.view((4, 3, 112, 112)).cpu().detach().numpy()

fig, axs = plt.subplots(2, 2)
axs[0, 0].imshow(x_np[0].swapaxes(0, 2).swapaxes(0, 1))
axs[0, 1].imshow(x_np[1].swapaxes(0, 2).swapaxes(0, 1))
axs[1, 0].imshow(x_np[2].swapaxes(0, 2).swapaxes(0, 1))
axs[1, 1].imshow(x_np[3].swapaxes(0, 2).swapaxes(0, 1))

plt.show()
