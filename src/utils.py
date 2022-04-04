import argparse
import torch
import torch.nn as nn

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
