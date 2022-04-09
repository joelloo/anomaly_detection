import os
import json
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
from utils import construct_parser, visualise

from pdb import set_trace as bp
# Require models to be saved to weights and biases

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

# Weights and biases
assert args.wandb_entity, "Please input weights and biases entity."
run = wandb.init(project="anomaly_detection", entity=args.wandb_entity, config=config)

# Download artifacts for all num of flows
flow_lengths = [2,16,32]
model_paths = []
for num_flows in flow_lengths:
    artifact = run.use_artifact(
        f'robot-anomaly-detection/anomaly_detection/vanilla-vae-e{args.load_epoch}-nf{num_flows}:latest',
        type='vae-models'
    )
    artifact_dir = artifact.download()
    model_paths.append(
        os.path.join(
            artifact_dir, 
            f'vae_v1_e{args.load_epoch}_nf{num_flows}.ckpt'
        )
    )

# Instantiate vae with normalising flow
n_bottleneck = config["n_bottleneck"]
b = torch.tensor(n_bottleneck // 2 * [0, 1] + n_bottleneck % 2 * [0])

# Reconstruction loss for thresholding
criterion = nn.MSELoss(reduction="none")

stats = dict()
for num_flows, model_path in zip(flow_lengths, model_paths):
    flows = []
    for i in range(num_flows):
        st_net = MLP(config["n_bottleneck"], 1024, 3)
        if i % 2 == 0:
            flows += [LatentMaskedAffineCoupling(b, st_net)]
        else:
            flows += [LatentMaskedAffineCoupling(1 - b, st_net)]

    prior = torch.distributions.normal.Normal(loc=0.0, scale=1.0)
    nf = NormalisingFlow(flows, prior, device, first_linear=None).to(device)
    model = VariationalAutoencoderResNet(device, flows=nf, latent_size=256, img_height=112, net_type='resnet18')


    model.load_state_dict(torch.load(model_path, map_location=device))
    
    losses = {'ood':[], 'ind':[]}
    for batch, data in tqdm(enumerate(test_loader), total=len(test_loader)):
        img, label = data
        img = img.to(device)

        recon_img, _, _, _ = model(img)
        loss = criterion(recon_img, img)

        losses[label[0]].append(loss.mean().item())

    # Select 4 images as reconstruction
    indices = np.random.randint(len(test_data), size=4)
    print("Images selected from: ", indices)

    test_ims = []
    for index in indices:
        img, label = test_data[index]
        test_ims.append(img)

    test_ims = torch.stack(test_ims).to(device)
    recon_ims, _, _, _ = model(test_ims)

    test_ims_np = test_ims.cpu().detach().permute(0,2,3,1).numpy()
    recon_ims_np = recon_ims.cpu().detach().permute(0,2,3,1).numpy()

    fig, axs = plt.subplots(4, 3)
    visualise(axs, test_ims_np, recon_ims_np)
    plt.savefig(f"images/vae-nf{num_flows}.png")

    ood_mean = np.mean(losses['ood'])
    ood_std = np.std(losses['ood'])
    ind_mean = np.mean(losses['ind'])
    ind_std = np.std(losses['ind'])

    stats[num_flows] = {'ood': (ood_mean, ood_std), 'ind': (ind_mean, ind_std)}

with open('stats.json', 'w') as fp:
    json.dump(stats, fp)

# Produce classification results over a range of thresholds (based on standard deviation from mean)
classification_results = dict()
for n_flows in stats.keys():
    model_stats = stats[n_flows]
    ood_mean, ood_std = model_stats['ood']

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

fig, ax = plt.subplots(1,2)
for num_flows in flow_lengths:
    success_rates = classification_results[num_flows]
    ood_success_rate = [success[0] for success in success_rates]
    ind_success_rate = [success[1] for success in success_rates]
    ax[0].plot(np.linspace(0.1, 1.5, 15), ood_success_rate, label=f"nf-{num_flows}")
    ax[1].plot(np.linspace(0.1, 1.5, 15), ind_success_rate, label=f"nf-{num_flows}")

ax[0].legend()
ax[0].set_xlabel("Number of Std Dev")
ax[0].set_ylabel("Accuracy")
ax[0].title.set_text("OOD")
ax[1].legend()
ax[1].set_xlabel("Number of Std Dev")
ax[1].title.set_text("IND")

plt.show()