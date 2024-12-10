import torch
from utils import save_images_conditional, save_images_diverse, get_constants
from models import DiffUNet
from diffusers import UNet2DModel
from diffusers import DDPMScheduler
from torchvision import transforms
from torch.utils.data import DataLoader
from datasets import PairedDataset

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="A script to parse training parameters.")
    
    # Adding arguments
    parser.add_argument("--dataset", default="IllustrisTNG", type=str, required=False, help="IllustrisTNG, Astrid, or SIMBA.")
    parser.add_argument("--condition", default="star", type=str, required=False, help="star or noise")
    parser.add_argument("--dir", default="100diff", type=str, required=False, help="100diff or 100same")  
    
    # Parsing arguments
    return parser.parse_args()

def sample(args):
    device = "cuda"

    model_path = f"/groups/mlprojects/dm_diffusion/final_conditional/{args.dataset}/{args.condition}/train_results/model_epoch_10.pt"
    config = {
            "epochs": 10,
            "batch_size": 4,
            "learning_rate": 1e-4,
            "num_timesteps": 1000,
            "image_size": 256,
            "conditioning_channels": 1,
            "target_channels": 1,
            "stellar_file": f'/groups/mlprojects/dm_diffusion/data/Maps_Mstar_{args.dataset}_LH_z=0.00.npy',
            "g_maps": f'/groups/mlprojects/dm_diffusion/data/{args.dataset}_Gravitational_Lensing/.',
            "dm_file": f'/groups/mlprojects/dm_diffusion/data/Maps_Mcdm_{args.dataset}_LH_z=0.00.npy',
            "gas_file": f"/groups/mlprojects/dm_diffusion/data/Maps_Mgas_{args.dataset}_LH_z=0.00.npy",
            "conditional": True, 
            "sigma_noise": 0,
            "num_workers": 1
        }

    model = DiffUNet(config, conditional=True).to(device)
    model.load_state_dict(torch.load(model_path))

    constants = get_constants()

    transform_dm = transforms.Compose([
        transforms.Lambda(lambda x: torch.log10(x + 1e-8)),  # Log transformation
        transforms.Lambda(lambda x: (x - constants['dm_mean']) / constants['dm_std'])
    ])

    transform_stellar = transforms.Compose([
        transforms.Lambda(lambda x: torch.log10(x + 1e-8)),  # Log transformation
        transforms.Lambda(lambda x: (x - constants['stellar_mean']) / constants['stellar_std'])
    ])

    filenames = [f"/groups/mlprojects/dm_diffusion/data/Maps_Mcdm_{args.dataset}_LH_z=0.00.npy"]
    transform = [transform_dm]
    filenames.append(f'/groups/mlprojects/dm_diffusion/data/Maps_Mstar_{args.dataset}_LH_z=0.00.npy')
    transform.append(transform_stellar)

    paired_dataset = PairedDataset(filenames, transform=transform)
    PairedDataloader = DataLoader(paired_dataset, batch_size=4, shuffle=True, num_workers = 1)

    noise_scheduler = DDPMScheduler(num_train_timesteps=1000, clip_sample=False)

    if args.dir == "100diff":
        save_images_conditional(model, noise_scheduler, f"/groups/mlprojects/dm_diffusion/final_conditional/{args.dataset}/{args.condition}/{args.dir}", PairedDataloader)
    elif args.dir == "100same":
        save_images_diverse(model, noise_scheduler, f"/groups/mlprojects/dm_diffusion/final_conditional/{args.dataset}/{args.condition}/{args.dir}", PairedDataloader)

if __name__ == "__main__":
    args = parse_args()
    sample(args)