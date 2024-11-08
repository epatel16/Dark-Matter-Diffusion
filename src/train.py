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
from datasets import SlicedDataset, NPYDataset
from utils import sample, sample_v1, get_constants
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
    parser.add_argument("--out_path", type=str, required=False, default="/groups/mlprojects/dm_diffusion/model_out/", help="Path to save models.")
    
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
        "conditioning_channels": 1,
        "target_channels": 1,
        "stellar_file": f'/groups/mlprojects/dm_diffusion/data/Maps_Mstar_{args.dataset}_LH_z=0.00.npy',
        "dm_file": f"/groups/mlprojects/dm_diffusion/data/Maps_Mcdm_{args.dataset}_LH_z=0.00.npy",
        "conditional": args.conditional
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(args.out_path, os.path.join(f"lr{args.learning_rate}_step{args.timesteps}_size{args.img_size}_cond{args.conditional}",timestamp))
    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, 'config.yaml'), "w") as file: yaml.dump(config, file, default_flow_style=False)
    # TODO: Save config to file
    
    log_transform = transforms.Lambda(
        lambda x: torch.log10(x + 1)  # Applying log transformation
    )

    # norm_transform = transforms.Lambda( # Using transform from -1 to 1
    #     lambda x: ((x - x.min()) / (x.max() - x.min()) - 0.5)*2
    # )

    norm_transform = transforms.Lambda( # Using transform from -1 to 1
        lambda x: ((x - x.min()) / (x.max() - x.min()))
    )

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Resize((args.img_size, args.img_size)),
                                    # transforms.RandomCrop((args.img_size, args.img_size)),
                                    log_transform,
                                    norm_transform, 
    ])

    dm_dataset = SlicedDataset(config["dm_file"], transform=transform)
    dm_dataloader = DataLoader(dm_dataset, batch_size=args.batch_size, shuffle=True)
    stellar_dataloader = None
    if args.conditional:
        stellar_dataset = SlicedDataset(config["stellar_file"], transform=transform)
        stellar_dataloader = DataLoader(stellar_dataset, batch_size=args.batch_size, shuffle=True)
    # pdb.set_trace()
    
    noise_scheduler = DDPMScheduler(num_train_timesteps=args.timesteps)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = DiffUNet(config, conditional=args.conditional).to(device)
    print(model.conditional)
    
    # Prepare everything with Accelerator
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    model, optimizer, dm_dataloader, stellar_dataloader = accelerator.prepare(model, optimizer, dm_dataloader, stellar_dataloader)

    epoch_losses = []
    for epoch in range(config["epochs"]):
        vis = True
        total_loss = 0.0
        num_batches = 0

        ep_dir = os.path.join(save_path, f'ep{epoch}')
        os.makedirs(ep_dir, exist_ok=True)

        # Set up progress bar for the current epoch
        if args.conditional:
            combined_maps = zip(stellar_dataloader, dm_dataloader)
            batch_progress = tqdm(combined_maps, desc=f"Epoch {epoch+1}/{config['epochs']}", leave=False)
        else:
            combined_maps = dm_dataloader
            batch_progress = tqdm(combined_maps, desc=f"Epoch {epoch+1}/{config['epochs']}", leave=False)
        
        # Batch processing within epoch
        for maps in batch_progress:
            if args.conditional:
                stellar_maps, dm_maps = maps[0].to(device), maps[1].to(device)
            else:
                dm_maps = maps.to(device)

            noise = torch.randn_like(dm_maps)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (dm_maps.shape[0],), device=device).long()
            noisy_dm = noise_scheduler.add_noise(dm_maps, noise, timesteps)
            
            if vis:
                fig = plt.figure()
                sample_img = plt.imshow(dm_maps[0][0].cpu().numpy(), cmap='viridis')
                fig.colorbar(sample_img)
                plt.savefig(os.path.join(ep_dir, "GT.png"))

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
        sample_v1(model, noise_scheduler, ep_dir, condition_loader=stellar_dataloader, device=device)

    # Save all epoch losses
    np.save(f'{save_path}/losses.npy', np.array(epoch_losses))

if __name__ == "__main__":
    args = parse_args()
    main(args)
