import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
import torchvision
from torchvision import transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from diffusers import UNet2DModel
from diffusers import DDPMScheduler

from tqdm import tqdm
from datasets import SlicedDataset, PairedDataset, NPYDataset
from utils import get_constants, add_gaussian_noise, blackout_pixels, sample_all
from accelerate import Accelerator
from torch.optim.lr_scheduler import CosineAnnealingLR
from models import DiffUNet_MultiCond

from datetime import datetime
import os
import argparse
import pdb
import yaml



def parse_args():
    parser = argparse.ArgumentParser(description="Parsing training parameters.")
    
    # Adding arguments
    parser.add_argument("--dataset", default="Astrid", type=str, required=False, help="IllustrisTNG, Astrid, or SIMBA.")
    parser.add_argument("--epochs", type=int, required=False, default= 30, help="Number of epochs for training.")
    parser.add_argument("--batch_size", type=int, required=False, default=12, help="Batch size for training.")
    parser.add_argument("--timesteps", type=int, required=False, default=1000, help="Timesteps for training.")
    parser.add_argument("--learning_rate", type=float, default=1e-4,required=False, help="Learning rate for training.")
    parser.add_argument("--img_size", type=int, required=False, default=256, help="Image size. Single int, (H = W).")
    parser.add_argument("--unconditional", action="store_true", help="Enable unconditional mode")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    # parser.add_argument("--out_path", type=str, required=False, default="model_out/", help="Path to save models.")
    parser.add_argument("--out_path", type=str, required=False, default="/groups/mlprojects/dm_diffusion/model_out/FinalExperiments/", help="Path to save models.")
    # Experiments
    parser.add_argument("--sigma_noise_lensing", type=float, required=False, default=0, help="Noise for lensing noise experiment.")
    parser.add_argument("--sigma_noise_stellar", type=float, required=False, default=0, help="Noise for mstar noise experiment.")
    parser.add_argument("--perc_preserved_frb", type=float, required=False, default=10, help="Percent mgas preserved (starting w/ 10% to simulate fast radio bursts)")
    parser.add_argument("--no_stellar", action="store_true", help="Disable Stellar")
    parser.add_argument("--no_frb", action="store_true", help="Disable FRB")
    parser.add_argument("--no_lensing", action="store_true", help="Disable Lensing")
    
    parser.add_argument("--exp_name", type=str, required=True, help="Experiment name")
    
    # Parsing arguments
    return parser.parse_args()


def main(args):
    accelerator = Accelerator()
    print(f"Using device: {accelerator.device}")
    args.conditional = not args.unconditional
    config = {
        "dataset": args.dataset,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "num_timesteps": args.timesteps,
        "image_size": args.img_size,
        "conditioning_channels": 0, # Add one w/ every new dataset
        "target_channels": 1,
        "stellar_file": f'/groups/mlprojects/dm_diffusion/data/Maps_Mstar_{args.dataset}_LH_z=0.00.npy',
        "dm_file": f"/groups/mlprojects/dm_diffusion/data/Maps_Mcdm_{args.dataset}_LH_z=0.00.npy",
        "gas_file": f"/groups/mlprojects/dm_diffusion/data/Maps_Mgas_{args.dataset}_LH_z=0.00.npy",
        "lensing1_file": f"/groups/mlprojects/dm_diffusion/data/Maps_Lensing1_{args.dataset}_LH_z=0.00.npy",
        "lensing2_file": f"/groups/mlprojects/dm_diffusion/data/Maps_Lensing2_{args.dataset}_LH_z=0.00.npy", 
        "conditional": args.conditional, 
        "sigma_noise_stellar": args.sigma_noise_stellar,
        "sigma_noise_lensing": args.sigma_noise_lensing,
        "perc_preserved_frb": args.perc_preserved_frb,
        "num_workers": 1,
        "target": "dm",
        "exp_name": args.exp_name
    }
    config["stellar"] = not args.no_stellar
    config["frb"] = not args.no_frb
    config["lensing"] = not args.no_lensing
    

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.debug: # Enable debug mode, send all runs to debug folder
        args.out_path = os.path.join(args.out_path, 'DEBUG')
    save_path = os.path.join(args.out_path, os.path.join(f"lr{args.learning_rate}_step{args.timesteps}_size{args.img_size}_cond{args.conditional}",config['exp_name']))
    save_path = os.path.join(save_path, timestamp)
    os.makedirs(save_path, exist_ok=True)

    constants = get_constants(dataset_name = args.dataset)

    transform_dm = transforms.Compose([
        transforms.Lambda(lambda x: torch.log10(x + 1e-8)),  # Log transformation
        transforms.Resize((config["image_size"], config["image_size"])),  
        transforms.Lambda(lambda x: (x - constants['dm_mean']) / constants['dm_std'])
    ])
    
    transform_stellar = transforms.Compose([
        transforms.Lambda(lambda x: torch.log10(x + 1e-8)),  # Log transformation
        transforms.Resize((config["image_size"], config["image_size"])), 
        transforms.Lambda(lambda x: (x - constants['stellar_mean']) / constants['stellar_std'])
    ])
    
    transform_gas = transforms.Compose([
        transforms.Lambda(lambda x: torch.log10(x + 1e-8)),  # Log transformation
        transforms.Resize((config["image_size"], config["image_size"])),  # Resize to specified size
        transforms.Lambda(lambda x: (x - constants['gas_mean']) / constants['gas_std']),
        transforms.Lambda(lambda img: blackout_pixels(img, percentage=config['perc_preserved_frb'])),  # Blackout 90% of pixels
    ])
    
    transform_lens1 = transforms.Compose([
        transforms.Resize((config["image_size"], config["image_size"])),  
        transforms.Lambda(lambda x: (x - constants['lens1_mean']) / constants['lens1_std']),
    ])
    
    transform_lens2 = transforms.Compose([
        transforms.Resize((config["image_size"], config["image_size"])), 
        transforms.Lambda(lambda x: (x - constants['lens2_mean']) / constants['lens2_std']),
    ])


    # dm_dataset = SlicedDataset(config["dm_file"], transform=transform)
    # dm_dataloader = DataLoader(dm_dataset, batch_size=config["batch_size"], shuffle=False)
    filenames = [config["dm_file"]]
    transform = [transform_dm]
    if args.conditional:
        if config["stellar"]:
            filenames.append(config["stellar_file"])
            transform.append(transform_stellar)
            config["conditioning_channels"] += 1
        if config["frb"]:
            filenames.append(config["gas_file"])
            transform.append(transform_gas)
            config["conditioning_channels"] += 1 
        if config["lensing"]:
            filenames.append(config["lensing1_file"])
            transform.append(transform_lens1)
            config["conditioning_channels"] += 1 
            filenames.append(config["lensing2_file"])
            transform.append(transform_lens2)
            config["conditioning_channels"] += 1 

    paired_dataset = PairedDataset(filenames, transform=transform)
    PairedDataloader = DataLoader(paired_dataset, batch_size=config["batch_size"], shuffle=True, num_workers = config["num_workers"])

    # Initialize the noise scheduler
    noise_scheduler = DDPMScheduler(num_train_timesteps=config["num_timesteps"], clip_sample=False)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = DiffUNet_MultiCond(config, conditional=args.conditional).to(device)
    
    # Prepare everything with Accelerator
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    model, optimizer, PairedDataloader = accelerator.prepare(model, optimizer, PairedDataloader)

    # Save configs
    with open(os.path.join(save_path, 'config.yaml'), "w") as file: yaml.dump(config, file, default_flow_style=False)
    print(model.config) # Verbose

    epoch_losses = []
    for epoch in range(config["epochs"]):
        vis = True
        total_loss = 0.0
        num_batches = 0

        ep_dir = os.path.join(save_path, f'ep{epoch}')
        os.makedirs(ep_dir, exist_ok=True)

        batch_progress = tqdm(PairedDataloader, desc=f"Epoch {epoch+1}/{config['epochs']}", leave=False)
        
        # Batch processing within epoch
        for maps in batch_progress:
            dm_maps = maps[0].to(device)
            if args.conditional:
                conditions = []
                index = 1
                if config["stellar"]:
                    stellar_maps = add_gaussian_noise(maps[index].to(device), sigma=config["sigma_noise_stellar"]) # adding gaussian noise in log transformed space, poisson noise in original space
                    index += 1
                    conditions.append(stellar_maps)
                if config["frb"]:
                    gas_maps = maps[index].to(device)
                    index += 1
                    conditions.append(gas_maps)
                if config["lensing"]:
                    lens_maps1 =  add_gaussian_noise(maps[index].to(device), sigma=config["sigma_noise_lensing"])
                    index += 1
                    conditions.append(lens_maps1)
                    lens_maps2 = add_gaussian_noise(maps[index].to(device), sigma=config["sigma_noise_lensing"])
                    index += 1
                    conditions.append(lens_maps2)
            
            noise = torch.randn_like(dm_maps).to(device)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (dm_maps.shape[0],), device=device).long()
            noisy_dm = noise_scheduler.add_noise(dm_maps, noise, timesteps) # we add noise to the target

            # Model prediction
            if args.conditional:
                pred_noise = model(noisy_dm, timesteps, conditions) # add gas maps here
            else:
                pred_noise = model(noisy_dm, timesteps)

            # Compute loss and backpropagation
            loss = nn.MSELoss()(pred_noise, noise)
            epoch_losses.append(loss.item())

            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()

            # Update progress bar with batch loss
            total_loss += loss.item()
            num_batches += 1
            batch_progress.set_postfix(batch_loss=loss.item())  # Shows current batch loss

        scheduler.step()

        # Average loss for the epoch
        avg_loss = total_loss / num_batches
        tqdm.write(f"Epoch [{epoch+1}/{config['epochs']}], Loss: {avg_loss:.4f}")

        # Save model checkpoint
        unwrapped_model = accelerator.unwrap_model(model)
        model_save_path = os.path.join(save_path, f"model_epoch_{epoch+1}.pt")
        torch.save(unwrapped_model.state_dict(), model_save_path)
        print(f"Model saved to {model_save_path}")

        # Sampling step
        sample_all(model, noise_scheduler, ep_dir, loader=PairedDataloader, config=config, conditional=config["conditional"], device=device)

    # Save all epoch losses
    np.save(f'{save_path}/losses.npy', np.array(epoch_losses))

if __name__ == "__main__":
    args = parse_args()
    main(args)
