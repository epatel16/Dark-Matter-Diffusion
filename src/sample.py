import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
import torchvision
from torchvision import transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from datasets import SlicedDataset, PairedDataset, NPYDataset

from diffusers import DDPMScheduler
from diffusers import UNet2DModel
from models import DiffUNet

from tqdm import tqdm
from datasets import SlicedDataset, NPYDataset
from utils import sample, get_constants, add_gaussian_noise
from models import DarkMatterUNet

from datetime import datetime
import os
import argparse
import yaml

def parse_args():
    parser = argparse.ArgumentParser(description="A script to parse training parameters.")
    
    # Adding arguments
    parser.add_argument("--dataset", default="IllustrisTNG", type=str, required=False, help="IllustrisTNG, Astrid, or SIMBA.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to specific model.")
    parser.add_argument("--batch_size", type=int, required=False, default=12, help="Batch size for training.")
    parser.add_argument("--idx", type=int, required=False, default=None, help="Specific index in dataset of sample.")
    
    
    # Parsing arguments
    return parser.parse_args()


def main(args):
    ep = args.model_path.split('_')[-1].split('.')[0]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # e.g. /groups/mlprojects/dm_diffusion/model_out/lr0.0001_step1000_size64_condTrue/20241118_182815/model_epoch_10.pt
    model_path = args.model_path
    config_path = os.path.dirname(model_path)
    folder_path = os.path.join(config_path, f"ep{ep}")
    
    with open(os.path.join(config_path, "config.yaml"), "r") as file:
        config = yaml.safe_load(file)
    
    print(config)

    constants = get_constants()

    transform_dm = transforms.Compose([
        transforms.Lambda(lambda x: torch.log10(x + 1e-8)),  # Log transformation
        transforms.Resize((config["image_size"], config["image_size"])),  # Resize to 64x64
        transforms.Lambda(lambda x: (x - constants['dm_mean']) / constants['dm_std'])
    ])
    
    transform_stellar = transforms.Compose([
        transforms.Lambda(lambda x: torch.log10(x + 1e-8)),  # Log transformation
        transforms.Resize((config["image_size"], config["image_size"])),  # Resize to 64x64
        transforms.Lambda(lambda x: (x - constants['stellar_mean']) / constants['stellar_std'])
    ])

    filenames = [config["dm_file"]]
    transform = [transform_dm]
    if config["conditional"]:
        filenames.append(config["stellar_file"])
        transform.append(transform_stellar)
        config["conditioning_channels"] = 1 

    paired_dataset = PairedDataset(filenames, transform=transform)
    PairedDataloader = DataLoader(paired_dataset, batch_size=config["batch_size"], shuffle=True, num_workers = config["num_workers"])

    noise_scheduler = DDPMScheduler(num_train_timesteps=config["num_timesteps"], clip_sample=False)
    
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sample_path = os.path.join(folder_path, f'sampling_{timestamp}')
    os.makedirs(sample_path, exist_ok=True)

    
    
    model = DiffUNet(config, conditional=config["conditional"]).to(device)
    model.to(device)
    model.load_state_dict(torch.load(model_path))
    sample(model, noise_scheduler, sample_path, loader=PairedDataloader, conditional=config["conditional"], device=device, idx=args.idx, sigma_noise=config["sigma_noise"])
    print(f"sample succesfully saved in {sample_path}")


    
if __name__ == "__main__":
    args = parse_args()
    main(args)
    
'''
Load the model
set args for this
Load the contstants
call sample 
import model
'''    
