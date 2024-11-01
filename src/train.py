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

from datetime import datetime
import os
import argparse
import pdb



def parse_args():
    parser = argparse.ArgumentParser(description="A script to parse training parameters.")
    
    # Adding arguments
    parser.add_argument("--dataset", default="IllustrisTNG", type=str, required=False, help="IllustrisTNG, Astrid, or SIMBA.")
    parser.add_argument("--epochs", type=int, required=False, default= 20, help="Number of epochs for training.")
    parser.add_argument("--batch_size", type=int, required=False, default=12, help="Batch size for training.")
    parser.add_argument("--learning_rate", type=float, default=1e-4,required=False, help="Learning rate for training.")
    parser.add_argument("--img_size", type=int, required=False, default=64, help="Image size. Single int, (H = W).")
    parser.add_argument("--out_path", type=str, required=False, default="/groups/mlprojects/dm_diffusion/model_out/", help="Path to save models.")
    
    # Parsing arguments
    return parser.parse_args()

# TODO: Move to config --> Work w/ 64x64, Test 256x256

def main(args):
    accelerator = Accelerator()
    print(f"Using device: {accelerator.device}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(args.out_path, f"{timestamp}")
    os.makedirs(save_path, exist_ok=True)

    npy_file = f"/groups/mlprojects/dm_diffusion/data/Maps_Mcdm_{args.dataset}_LH_z=0.00.npy"
    
    log_transform = transforms.Lambda(
        lambda x: torch.log10(x + 1)  # Applying log transformation
    )

    norm_transform = transforms.Lambda( # Using transform from -1 to 1
        lambda x: ((x - x.min()) / (x.max() - x.min()) - 0.5)*2
    )

    constants = get_constants(args.dataset)
    # Using transforms.Normalize((constants['mean'],), (constants['std'],)) where mean and std correspond 
    # to the dataset does not work

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Resize((args.img_size, args.img_size)),
                                    # transforms.RandomCrop((args.img_size, args.img_size)),
                                    log_transform,
                                    norm_transform, 
                                    # transforms.Normalize((constants['mean'],), (constants['std'],))
    ])

    dataset = SlicedDataset(npy_file, transform=transform)
    # pdb.set_trace()

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

    model = UNet2DModel(
        sample_size=args.img_size,                  
        in_channels=1,                    
        out_channels=1,                   
        down_block_types=(
            "DownBlock2D", "DownBlock2D", "DownBlock2D", "AttnDownBlock2D"
        ),                                
        up_block_types=(
            "AttnUpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"
        ),                                
        block_out_channels=(48, 96, 192, 384),  
        layers_per_block=2,               
        norm_num_groups=8,                
        act_fn="silu",                    
        # time_embedding_type="fourier",    # Breaks it
        attention_head_dim=8,             
        add_attention=True,               
        num_train_timesteps=1000,         
        downsample_type='resnet',
        upsample_type='resnet'
    )
    # Added resnet blocks instead of conv, fourier time embeddings, out channels for blocks
    
    # Prepare everything with Accelerator
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)


    losses = []
    for epoch in range(args.epochs):
        imgs = True
        with tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}", unit="batch") as tepoch:
            for step, image in enumerate(tepoch):
                if imgs: 
                    plt.figure()
                    plt.imshow(image[0,0,:,:].detach().cpu().numpy())
                    plt.savefig(os.path.join(save_path, f'GT_ep{epoch}.png'), format='png')
                    plt.close()
                    imgs = False
                clean_images = image.to(accelerator.device)
                # Sample noise to add to the images
                noise = torch.randn(clean_images.shape).to(clean_images.device)
                bs = clean_images.shape[0]

                # Sample random timestep for each image
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (bs,), device=clean_images.device
                ).long()

                # Add noise to the clean images according to the noise magnitude at each timestep
                noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

                noise_pred = model(noisy_images, timesteps)

                noise_pred = noise_pred[0] 
                loss = F.mse_loss(noise_pred, noise)
                accelerator.backward(loss)
                losses.append(loss.item())

                optimizer.step()
                optimizer.zero_grad()

                tepoch.set_postfix(loss=loss.item())

        scheduler.step()
        # Calculate and print the loss for the last epoch
        loss_last_epoch = sum(losses[-len(dataloader):]) / len(dataloader)
        print(f"Epoch {epoch+1} loss: {loss_last_epoch}")
        
        # Save the model after every epoch
        unwrapped_model = accelerator.unwrap_model(model)
        model_save_path = os.path.join(save_path, f"model_epoch_{epoch+1}.pt")
        torch.save(unwrapped_model.state_dict(), model_save_path)
        print(f"Model saved to {model_save_path}")

        # Sample
        sample_v1(model, noise_scheduler, os.path.join(save_path, f'ep{epoch}_sample.png'), device='cuda')

    # Save all losses
    losses = np.array(losses)
    np.save(f'{save_path}/losses.npy', losses)


if __name__ == "__main__":
    args = parse_args()
    main(args)
