import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
import torchvision
from torchvision import transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from diffusers import DDPMScheduler
from diffusers import UNet2DModel

from tqdm import tqdm
from datasets import SlicedDataset, PairedDataset, NPYDataset
from utils import sample, sample_v3, sample_v2, get_constants, add_gaussian_noise
from accelerate import Accelerator
from torch.optim.lr_scheduler import CosineAnnealingLR
from models import DiffUNet

from datetime import datetime
import os
import argparse
import pdb
import yaml



def parse_args():
    parser = argparse.ArgumentParser(description="A script to parse training parameters.")
    
    # Adding arguments
    parser.add_argument("--dataset", default="IllustrisTNG", type=str, required=False, help="IllustrisTNG, Astrid, or SIMBA.")
    parser.add_argument("--epochs", type=int, required=False, default= 10, help="Number of epochs for training.")
    parser.add_argument("--batch_size", type=int, required=False, default=12, help="Batch size for training.")
    parser.add_argument("--timesteps", type=int, required=False, default=1000, help="Timesteps for training.")
    parser.add_argument("--learning_rate", type=float, default=1e-4,required=False, help="Learning rate for training.")
    parser.add_argument("--img_size", type=int, required=False, default=64, help="Image size. Single int, (H = W).")
    parser.add_argument("--unconditional", action="store_true", help="Enable unconditional mode")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--out_path", type=str, required=False, default="/groups/mlprojects/dm_diffusion/model_out/noised_mstar/", help="Path to save models.")
    
    # Parsing arguments
    return parser.parse_args()


def main(args):
    accelerator = Accelerator()
    print(f"Using device: {accelerator.device}")
    args.conditional = not args.unconditional
    config = {
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "num_timesteps": args.timesteps,
        "image_size": args.img_size,
        "conditioning_channels": 0, # Add one w/ every new dataset
        "target_channels": 1,
        "stellar_file": f'/groups/mlprojects/dm_diffusion/data/Maps_Mstar_{args.dataset}_LH_z=0.00.npy',
        "dm_file": f"/groups/mlprojects/dm_diffusion/data/Maps_Mcdm_{args.dataset}_LH_z=0.00.npy",
        "conditional": args.conditional, 
        "sigma_noise": 0,
        "num_workers": 1
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.debug: # Enable debug mode, send all runs to debug folder
        args.out_path = os.path.join(args.out_path, 'DEBUG')
    save_path = os.path.join(args.out_path, os.path.join(f"lr{args.learning_rate}_step{args.timesteps}_size{args.img_size}_cond{args.conditional}",timestamp))
    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, 'config.yaml'), "w") as file: yaml.dump(config, file, default_flow_style=False)

    # transform = transforms.Compose([
    #     transforms.Lambda(lambda x: torch.log10(x + 1e-8)),  # Log transformation
    #     transforms.Resize((config["image_size"], config["image_size"])),  # Resize to 64x64
    #     transforms.Lambda(lambda x: ((x - x.min()) / (x.max() - x.min()) - 0.5) * 2)  # Normalize to [-1, 1]
    # ])

    transform_dm = transforms.Compose([
        transforms.Lambda(lambda x: torch.log10(x + 1e-8)),  # Log transformation
        transforms.Resize((config["image_size"], config["image_size"])),  # Resize to 64x64
        transforms.Lambda(lambda x: (x - 10.971004486083984) / 0.5090954303741455)
        # transforms.Normalize((10.971004486083984,), (0.5090954303741455,)) # Global Normalization DM: 10.971004486083984, 0.5090954303741455
    ])
    
    transform_stellar = transforms.Compose([
        transforms.Lambda(lambda x: torch.log10(x + 1e-8)),  # Log transformation
        transforms.Resize((config["image_size"], config["image_size"])),  # Resize to 64x64
        transforms.Lambda(lambda x: (x - 0.11826974898576736) / 1.0741989612579346)
        # transforms.Normalize((0.11826974898576736,), (1.0741989612579346,)) # Global Normalization Stellar: 0.11826974898576736, 1.0741989612579346,
    ])

    # dm_dataset = SlicedDataset(config["dm_file"], transform=transform)
    # dm_dataloader = DataLoader(dm_dataset, batch_size=config["batch_size"], shuffle=False)
    filenames = [config["dm_file"]]
    transform = [transform_dm]
    if args.conditional:
        filenames.append(config["stellar_file"])
        transform.append(transform_stellar)
        config["conditioning_channels"] = 1 # Change this as we add modalities

    paired_dataset = PairedDataset(filenames, transform=transform)
    PairedDataloader = DataLoader(paired_dataset, batch_size=config["batch_size"], shuffle=True, num_workers = config["num_workers"])
    # stellar_dataloader = None
    # if args.conditional:
    #     # stellar_dataset = SlicedDataset(config["stellar_file"], transform=transform)
    #     stellar_dataloader = DataLoader(stellar_dataset, batch_size=config["batch_size"], shuffle=False)
    #     config["conditioning_channels"] = 1 # Change this as we add modalities

    # Initialize the noise scheduler
    noise_scheduler = DDPMScheduler(num_train_timesteps=config["num_timesteps"], clip_sample=False)


    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = DiffUNet(config, conditional=args.conditional).to(device)
    print(model.conditional)
    
    # Prepare everything with Accelerator
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    model, optimizer, PairedDataloader = accelerator.prepare(model, optimizer, PairedDataloader)

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
                stellar_maps = add_gaussian_noise(maps[1].to(device), sigma=config["sigma_noise"]) # adding gaussian noise in log transformed space, poisson noise in original space

            noise = torch.randn_like(dm_maps).to(device)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (dm_maps.shape[0],), device=device).long()
            noisy_dm = noise_scheduler.add_noise(dm_maps, noise, timesteps)

            # Model prediction
            if args.conditional:
                pred_noise = model(noisy_dm, timesteps, stellar_maps)
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

            if vis:
                pred_vis = (pred_noise-noise)
                fig, axes = plt.subplots(1, 3, figsize=(18, 6))
                sample_img_gt = axes[0].imshow(dm_maps[0][0].cpu().numpy(), cmap='viridis')
                axes[0].set_title("Ground Truth")
                axes[0].axis('off')
                fig.colorbar(sample_img_gt, ax=axes[0], fraction=0.046, pad=0.04)
                sample_img_stellar = axes[1].imshow(stellar_maps[0][0].cpu().numpy(), cmap='viridis')
                axes[1].set_title("Stellar Map")
                axes[1].axis('off')
                fig.colorbar(sample_img_stellar, ax=axes[1], fraction=0.046, pad=0.04)
                sample_img_estimated = axes[2].imshow(pred_vis[0][0].detach().cpu().numpy(), cmap='viridis')
                axes[2].set_title("Estimated DM")
                axes[2].axis('off')
                fig.colorbar(sample_img_estimated, ax=axes[2], fraction=0.046, pad=0.04)
                plt.tight_layout()
                plt.savefig(os.path.join(ep_dir, "comparison.png"))
                plt.close(fig)

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
        sample_v3(model, noise_scheduler, ep_dir, loader=PairedDataloader, conditional=config["conditional"], device=device)

    # Save all epoch losses
    np.save(f'{save_path}/losses.npy', np.array(epoch_losses))

if __name__ == "__main__":
    args = parse_args()
    main(args)
