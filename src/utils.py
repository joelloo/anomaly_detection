import cv2
import argparse
import torch
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt

from pdb import set_trace as bp

def construct_parser():
    """ Argument parsers """
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, help="Path to dataset",
        default="data/train")
    parser.add_argument("--dataset_type", type=str, help="Pick which dataset to use: left, middle, right, all",
        default="all")
    parser.add_argument("--model_dir", type=str, help="Path to save models",
        default="trained_models")
    parser.add_argument("--wandb_entity", "-e", type=str, help="Weights and Biases entity",
        default=None)
    return parser

def find_threshold(model, dataset, device, model_type):
    """ Determine anomaly threshold on a dataset """
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=32, 
        shuffle=True, drop_last=False,
    )

    criterion = nn.MSELoss(reduction="none")
    x = next(iter(data_loader))
    x = x.to(device)
    # Just a random sample of all images, else cuda will ooom
    if model_type == "vae":
        recon_x, _, _, _ = model(x)
    else:
        recon_x = model(x)

    loss = criterion(recon_x, x)

    threshold_mean = loss.mean().item()
    threshold_std = loss.std().item()
    return threshold_mean, threshold_std

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