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
from models import DiffUNet_MultiCond

from tqdm import tqdm
from datasets import SlicedDataset, NPYDataset
from utils import sample_multiple_ood, get_constants, add_gaussian_noise, blackout_pixels

from datetime import datetime
import os
import argparse
import yaml

def parse_args():
    parser = argparse.ArgumentParser(description="A script to parse training parameters. \
        Note, model must have already been trained with any modality you wish to condition on.")
    
    # Adding arguments
    parser.add_argument("--dataset", type=str, required=False, default='Astrid', help="Path to specific model.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to specific model.")
    parser.add_argument("--batch_size", type=int, required=False, default=12, help="Batch size for training.")
    parser.add_argument("--idx", type=int, required=False, default=None, help="Specific index in dataset of sample.")
    parser.add_argument("--N", type=int, required=False, default=10, help="Number of samples to generate.")
    parser.add_argument("--out_path", type=str, required=False, default='/groups/mlprojects/dm_diffusion/sample_ood/', help="Path to save to")
    # parser.add_argument("--out_path", type=str, required=False, default='sample_ood/', help="Path to save to")


    parser.add_argument("--stellar", action="store_true", help="To condition on stellar")
    parser.add_argument("--frb", action="store_true", help="To condition on frb")
    parser.add_argument("--lensing", action="store_true", help="To condition on lensing")
    
    # Parsing arguments
    return parser.parse_args()


def main(args):
    ep = args.model_path.split('_')[-1].split('.')[0]
    model_path = args.model_path
    config_path = os.path.dirname(model_path)
    folder_path = os.path.join(config_path, f"ep{ep}")
    with open(os.path.join(config_path, "config.yaml"), "r") as file:
        config = yaml.safe_load(file)
    config["sample_dataset"] = args.dataset # Dataset for Sampling
    config["sample_stellar"] = args.stellar 
    config["sample_frb"] = args.frb
    config["sample_lens"] = args.lensing
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    modalities = []
    for c, m in zip([args.stellar, args.frb, args.lensing], ['star', 'frb', 'lens']):
        if c: modalities.append(m)
    experiment_name = '-'.join(modalities)

    save_path = os.path.join(args.out_path, f'{args.dataset}_{experiment_name}')
    save_path = os.path.join(save_path, timestamp)
    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, 'config.yaml'), "w") as file: yaml.dump(config, file, default_flow_style=False)
    
    constants = get_constants(config["sample_dataset"])

    transform_dm = transforms.Compose([
        transforms.Lambda(lambda x: torch.log10(x + 1e-8)),  # Log transformation
        transforms.Resize((config["image_size"], config["image_size"])),  # Resize to 64x64
        transforms.Lambda(lambda x: (x - constants['dm_mean']) / constants['dm_std'])
    ])
    
    if config["sample_stellar"]:
        transform_stellar = transforms.Compose([
            transforms.Lambda(lambda x: torch.log10(x + 1e-8)),  # Log transformation
            transforms.Resize((config["image_size"], config["image_size"])),  # Resize to 64x64
            transforms.Lambda(lambda x: (x - constants['stellar_mean']) / constants['stellar_std'])
        ])

    if config["sample_frb"]:
        transform_gas = transforms.Compose([
            transforms.Lambda(lambda x: torch.log10(x + 1e-8)),  # Log transformation
            transforms.Resize((config["image_size"], config["image_size"])),  # Resize to specified size
            transforms.Lambda(lambda x: (x - constants['gas_mean']) / constants['gas_std']),
            transforms.Lambda(lambda img: blackout_pixels(img, percentage=config['perc_preserved_frb'])),  # Blackout 90% of pixels
        ])
    
    if config["sample_lens"]:
        transform_lens1 = transforms.Compose([
            transforms.Resize((config["image_size"], config["image_size"])),  # Resize to 64x64
            transforms.Lambda(lambda x: (x - constants['lens1_mean']) / constants['lens1_std']), # TODO: change to correct constants
        ])
        
        transform_lens2 = transforms.Compose([
            transforms.Resize((config["image_size"], config["image_size"])),  # Resize to 64x64
            transforms.Lambda(lambda x: (x - constants['lens2_mean']) / constants['lens2_std']), # TODO: change to correct constants
        ])

    filenames = [config["dm_file"]]
    transform = [transform_dm]
    if config["conditional"]:
        if config["sample_stellar"]:
            filenames.append(config["stellar_file"])
            transform.append(transform_stellar)
        if config["sample_frb"]:
            filenames.append(config["gas_file"])
            transform.append(transform_gas)
        if config["sample_lens"]:
            filenames.append(config["lensing1_file"])
            transform.append(transform_lens1)
            filenames.append(config["lensing2_file"])
            transform.append(transform_lens2)

    paired_dataset = PairedDataset(filenames, transform=transform)
    PairedDataloader = DataLoader(paired_dataset, batch_size=config["batch_size"], shuffle=True, num_workers = config["num_workers"])

    noise_scheduler = DDPMScheduler(num_train_timesteps=config["num_timesteps"], clip_sample=False)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = DiffUNet_MultiCond(config, conditional=config['conditional']).to(device)
    
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # sample_path = os.path.join(folder_path, f'sampling_{timestamp}')
    

    model.to(device)
    model.load_state_dict(torch.load(model_path))
    sample_multiple_ood(model, noise_scheduler, save_path, loader=PairedDataloader, config=config, constants=constants, conditional=config["conditional"], device=device, idx=args.idx, N=args.N)
    print(f"sample succesfully saved in {save_path}")


    
if __name__ == "__main__":
    args = parse_args()
    main(args)