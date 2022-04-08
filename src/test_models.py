import os
import torch
import torch.nn as nn
import wandb
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from datasets.testsets import TestDataSet
from models.vae import VariationalAutoencoderResNet
from models.realnvp import MLP, LatentMaskedAffineCoupling, NormalisingFlow
from datasets.datasets import load_all_datasets
from utils import construct_parser, find_threshold, generate_rmse_loss_heatmap

from pdb import set_trace as bp

# Parsing arguments
parser = construct_parser()
parser.add_argument("--train_data", type=str, help="Directory for train data",
        default="data/train")
parser.add_argument("--model_type", type=str, help="model_type",
        default="vae")
parser.add_argument("--load_epoch", type=int, help="Epoch of model to load", default=49)
args = parser.parse_args()

# Initialize device and dataset
device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')

# Create testset data loader
test_data = TestDataSet(data_dir=args.dataset_dir)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=True, drop_last=False)

# Create train data set
full_model_dir = os.path.join(os.getcwd(), args.model_dir)
datasets_map = load_all_datasets(os.path.join(os.getcwd(), args.train_data))
if args.dataset_type == "all":
    datasets_list = list(datasets_map.values())
    train_dataset = torch.utils.data.ConcatDataset(datasets_list)
else:
    train_dataset = datasets_map[args.dataset_type]

config = {
    "n_bottleneck": 256
}

# Weights and biases logging
if args.wandb_entity:
    run = wandb.init(project="anomaly_detection", entity=args.wandb_entity, config=config)

# Check if model choice is valid and load model
model_types = ("vae", "ae")
assert args.model_type in model_types, f"Model type should be {model_types}"

if args.model_type == "ae":
    # Instantiate a ResNet-based autoencoder
    from pl_bolts.models.autoencoders import AE
    model = AE(input_height=112, enc_type='resnet18').to(device)
elif args.num_flows == 0:
    # Instantiate vanilla vae
    model = VariationalAutoencoderResNet(device, flows=None, latent_size=256, img_height=112, net_type='resnet18')
else:
    # Instantiate vae with normalising flow
    n_bottleneck = config["n_bottleneck"]
    b = torch.tensor(n_bottleneck // 2 * [0, 1] + n_bottleneck % 2 * [0])
    flows = []
    for i in range(args.num_flows):
        st_net = MLP(config["n_bottleneck"], 1024, 3)
        if i % 2 == 0:
            flows += [LatentMaskedAffineCoupling(b, st_net)]
        else:
            flows += [LatentMaskedAffineCoupling(1 - b, st_net)]

    prior = torch.distributions.normal.Normal(loc=0.0, scale=1.0)
    nf = NormalisingFlow(flows, prior, device, first_linear=None).to(device)
    vae = VariationalAutoencoderResNet(device, flows=nf, latent_size=256, img_height=112, net_type='resnet18')

# Reconstruction loss for thresholding
criterion = nn.MSELoss(reduction="none")
    
# Testing the performance of each model epoch
model_path = os.path.join(args.model_dir, f"{args.model_type}_v1_e{args.load_epoch}_nf{args.num_flow}.ckpt")

# If model path don't exist, download from weights and biases
if not os.path.exists(os.path.join(os.getcwd(), model_path)):
    if args.wandb_entity:
        artifact = run.use_artifact(
            f'robot-anomaly-detection/anomaly_detection/{args.model_type}-e{args.load_epoch}:latest',
            type=f'{args.model_type}-models'
        )
        artifact_dir = artifact.download()
        model_path = os.path.join(artifact_dir, f'{args.model_type}_v1_e{args.load_epoch}_nf{args.num_flow}.ckpt')
    else:
        raise RuntimeError(f"Unable to find models from {model_path}")

model.load_state_dict(torch.load(model_path, map_location=device))

losses = {'ood':[], 'ind':[]}
for batch, data in tqdm(enumerate(test_loader), total=len(test_loader)):
    img, label = data
    img = img.to(device)

    if args.model_type == "vae":
        recon_img, _, _, _ = model(img)
    else:
        recon_img = model(img)

    bp()
    # Save every 50th image as reconstruction visualization
    if batch % 50 == 0:
        recon_img = recon_img.detach().squeeze(0).permute(1,2,0).cpu().numpy()
        img = img.detach().squeeze(0).permute(1,2,0).cpu().numpy()
        _, combine = generate_rmse_loss_heatmap(img, recon_img)
        plt.imsave(f"images/heatmaps/{args.model_type}/map-e{epoch}-{batch}.png", combine)
        try:
            plt.imsave(f"images/recon/{args.model_type}/recon-e{epoch}-{batch}.png", recon_img)
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

del model
torch.cuda.empty_cache()

