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

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", type=str, help="Path to models",
    default="/home/joel/source/saved_models/models")
parser.add_argument("--encoder_dir", type=str, help="Path to encoder weights",
    default="/home/joel/source/saved_models/ae/ae_v1_e49.ckpt")
parser.add_argument("--test_dir", type=str, help="Path to test data",
    default="/home/joel/test_data")

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

# Set up test dataset
test_data = TestDataSet(data_dir=args.test_dir)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False, drop_last=False)

models = glob.glob(os.path.join(args.model_dir, "*.ckpt"))

stats = dict()
for model in models:
    tokens = model.split("_ |.")
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
    nf.load_state_dict(torch.load(os.path.join(args.model_dir)))
    nf.to(device)
    nf.eval()

    losses = {'ood':[], 'ind':[]}
    for batch, data in tqdm(enumerate(test_loader), total=len(test_loader)):
        img, label = data
        img = img.to(device)
        label = label[0]

        z = ae.encoder(img)
        z = ae.fc(z)
        z, log_det = nf(z)
        log_prob_z = nf.get_prior_log_prob(z)
        loss = -log_prob_z.mean() - log_det.mean()
        losses[label].append(loss.cpu().item())


    ood_mean = np.mean(losses['ood'])
    ood_std = np.std(losses['ood'])
    ind_mean = np.mean(losses['ind'])
    ind_std = np.std(losses['ind'])

    stats[n_flows] = {'ood': (ood_mean, ood_std), 'ind': (ind_mean, ind_std)}

with open('stats.json', 'w') as fp:
    json.dump(stats, fp)

# Produce classification results over a range of thresholds (based on standard deviation from mean)
classification_results = dict()
for n_flows in stats.keys():
    model_stats = stats[n_flows]
    ood_mean, ood_std = stats['ood']

    success_rates = []
    for threshold in np.linspace(0.1, 1.5, 15):
        ood_loss_threshold = ood_mean - threshold * ood_std

        results = {'ood': {'pass': 0, 'fail': 0}, 'ind': {'pass': 0, 'fail': 0}}
        for label, loss_array in losses.items():
            for loss in loss_array:
                if loss > ood_loss_threshold:
                    if label == 'ood':
                        results['ood']['pass'] += 1
                    else:
                        results['ind']['fail'] += 1
                else:
                    if label == 'ood':
                        results['ood']['fail'] += 1
                    else:
                        results['ind']['pass'] += 1

        ood_success_rate = float(results['ood']['pass']) / float(results['ood']['pass'] + results['ood']['fail'])
        ind_success_rate = float(results['ind']['pass']) / float(results['ind']['pass'] + results['ind']['fail'])
        success_rates.append([ood_success_rate, ind_success_rate])

        print(f'{n_flows} flows -- OOD rate: {ood_success_rate}, IND rate: {ind_success_rate}')

    classification_results[n_flows] = success_rates

with open('classification.json', 'w') as fp:
    json.dump(classification_results, fp)