import os
import wandb
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from models.vae import VariationalAutoencoderResNet
from models.realnvp import LatentMaskedAffineCoupling, NormalisingFlow, MLP
from datasets.datasets import load_all_datasets
from utils import construct_parser

from pdb import set_trace as bp

parser = construct_parser()
args = parser.parse_args()

### Train convolutional variational autoencoder ###

# Set up model hyperparameters
train_partition_fraction = 1.0
identifier_str = 'vae_v1'

enable_cuda = True
device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() and enable_cuda else 'cpu')

# Load dataset
workdir = os.getcwd()
full_model_dir = os.path.join(workdir, args.model_dir)
datasets_map = load_all_datasets(os.path.join(workdir, args.dataset_dir))

config = {
  "learning_rate": 1e-4,
  "epochs": 50,
  "batch_size": 32,
  "weight_decay": 1e-4,
  "num_flows": args.num_flows,
  "n_bottleneck": 256
}

# Instantiate a variational autoencoder
# vae = VariationalAutoencoder(device).to(device)
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

# fc_adapter = nn.Linear(1000, n_bottleneck, bias=True)
nf = NormalisingFlow(flows, prior, device, first_linear=None).to(device)
vae = VariationalAutoencoderResNet(device, flows=nf, latent_size=256, img_height=224, net_type='resnet18')

# Weights and biases logging
if args.wandb_entity:
    wandb.init(
        id=f"train-vae-nf{args.num_flows}",
        project="anomaly_detection", 
        entity=args.wandb_entity,
        config=config,
        resume="allow"
    )

if args.dataset_type == "all":
    datasets_list = list(datasets_map.values())
    train_dataset = torch.utils.data.ConcatDataset(datasets_list)
else:
    train_dataset = datasets_map[args.dataset_type]

train_partition_len = int(np.floor(train_partition_fraction * len(train_dataset)))
val_partition_len = len(train_dataset) - train_partition_len
train_set, val_set = torch.utils.data.random_split(train_dataset, [train_partition_len, val_partition_len])

train_loader = torch.utils.data.DataLoader(train_set, batch_size=config["batch_size"], shuffle=True, drop_last=False)

print("Loaded data: ", args.dataset_type)
print("Train set: ", len(train_set))
print("Val set: ", len(val_set))

# Train the VAE
optimizer =  torch.optim.Adam(vae.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
recon_loss = nn.MSELoss(reduction='mean')

for epoch in range(config["epochs"]):
    progressbar = tqdm(enumerate(train_loader), total=len(train_loader))
    run_recon_loss = 0.0
    run_var_loss = 0.0

    for batch_n, x in progressbar:
        x = x.to(device)
        optimizer.zero_grad()
        outputs, mu, sigma, var_loss = vae(x)
        rloss = recon_loss(outputs, x)
        loss = rloss + var_loss
        loss.backward()
        optimizer.step()
        progressbar.update()

        run_recon_loss += rloss
        run_var_loss += var_loss

    progressbar.close()
    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_n * len(x), len(train_loader.dataset),
                       100. * batch_n / len(train_loader),
                       loss.item()))

    if args.wandb_entity:
        wandb.log({
            "avg_recon_loss": run_recon_loss / batch_n,
            "avg_var_loss": run_var_loss / batch_n
        }, step=epoch)
                      
    # Save for every epoch
    print(f'Saving checkpoint for epoch {epoch}...')
    torch.save(vae.state_dict(), os.path.join(full_model_dir, f'{identifier_str}_e{epoch}.ckpt'))
    if args.wandb_entity:
        wandb.log_artifact(os.path.join(full_model_dir, f'{identifier_str}_e{epoch}.ckpt'), name=f'vanilla-vae-e{epoch}', type='vae-models') 

# Save the final checkpoint
torch.save(vae.state_dict(), os.path.join(full_model_dir, f'{identifier_str}_e{epoch}.ckpt'))
