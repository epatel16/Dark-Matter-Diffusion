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
from utils import LogTransform
from sample import sample
from accelerate import Accelerator

from datetime import datetime
import os
import argparse

from models import DarkMatterUNet

def parse_args():
    parser = argparse.ArgumentParser(description="A script to parse training parameters.")
    
    # Adding arguments
    parser.add_argument("--dataset", default="IllustrisTNG", type=str, required=False, help="IllustrisTNG, Astrid, or SIMBA.")
    parser.add_argument("--epochs", type=int, required=False, default= 30, help="Number of epochs for training.")
    parser.add_argument("--batch_size", type=int, required=False, default=32, help="Batch size for training.")
    parser.add_argument("--learning_rate", type=float, default=4e-4,required=False, help="Learning rate for training.")
    parser.add_argument("--img_size", type=int, required=False, default=64, help="Image size. Single int, (H = W).")
    parser.add_argument("--out_path", type=str, required=False, default="/groups/mlprojects/dm_diffusion/model_out/", help="Path to save models.")
    
    # Parsing arguments
    return parser.parse_args()


def main(args):
    accelerator = Accelerator()
    print(f"Using device: {accelerator.device}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(args.out_path, f"{timestamp}")
    os.makedirs(save_path, exist_ok=True)

    npy_file = f"/groups/mlprojects/dm_diffusion/data/Maps_Mcdm_{args.dataset}_LH_z=0.00.npy"
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.RandomCrop((args.img_size, args.img_size)),
                                    LogTransform(),
                                    transforms.Normalize((0.0,), (1.0,))])

    dataset = SlicedDataset(npy_file, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    noise_scheduler = DDPMScheduler(num_train_timesteps=10000)

    # Create a model
    model = UNet2DModel( # TODO: pull this into a config
        sample_size=args.img_size,  # the target image resolution
        in_channels=1,  # the number of input channels, 3 for RGB images
        out_channels=1,  # the number of output channels
        layers_per_block=2,  # how many ResNet layers to use per UNet block
        block_out_channels=(64, 128, 256),  # the number of output channels for each UNet block
        down_block_types=(
            "DownBlock2D",  # a regular ResNet downsampling block
            "DownBlock2D",
            "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
        ),
        up_block_types=(
            "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
            "UpBlock2D",
            "UpBlock2D",
        ),
    )
    # model = DarkMatterUNet(in_channels=1, out_channels=1, base_channels=64)

    # model = UNet2DModel(
    #     sample_size=256,  # Image resolution
    #     in_channels=1,  # Number of input channels (e.g., grayscale images)
    #     out_channels=1,  # Number of output channels
    #     layers_per_block=2,  # Number of ResNet layers per UNet block
    #     block_out_channels=(64, 128, 256, 512),  # Channels for each block
    #     down_block_types=(
    #         "DownBlock2D",  # Regular downsampling block
    #         "AttnDownBlock2D",  # Downsampling block with spatial attention
    #         "DownBlock2D",
    #         "AttnDownBlock2D",
    #     ),
    #     up_block_types=(
    #         "AttnUpBlock2D",  # Upsampling block with spatial attention
    #         "UpBlock2D",
    #         "AttnUpBlock2D",
    #         "UpBlock2D",
    #     ),
    # )

    # Prepare everything with Accelerator
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    # Training loop
    losses = []
    for epoch in range(args.epochs):
        # Add tqdm progress bar for the epoch
        with tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}", unit="batch") as tepoch:
            for step, image in enumerate(tepoch):
                clean_images = image.to(accelerator.device)
                # Sample noise to add to the images
                noise = torch.randn(clean_images.shape).to(clean_images.device)
                bs = clean_images.shape[0]

                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (bs,), device=clean_images.device
                ).long()

                # Add noise to the clean images according to the noise magnitude at each timestep
                noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)
                # print('NOISY IMGS: ', noisy_images.shape)
                # print('CLEAN IMGS: ', clean_images.shape)
                # Get the model prediction
                noise_pred = model(noisy_images, timesteps)

                # Calculate the loss
                # print(noise_pred.shape, noise.shape)
                noise_pred = noise_pred[0] 
                loss = F.mse_loss(noise_pred, noise)
                accelerator.backward(loss)
                losses.append(loss.item())

                # Update the model parameters with the optimizer
                optimizer.step()
                optimizer.zero_grad()

                # Update tqdm with the current loss
                tepoch.set_postfix(loss=loss.item())

        # Calculate and print the loss for the last epoch
        loss_last_epoch = sum(losses[-len(dataloader):]) / len(dataloader)
        print(f"Epoch {epoch+1} loss: {loss_last_epoch}")
        
        # Save the model after every epoch
        unwrapped_model = accelerator.unwrap_model(model)
        model_save_path = os.path.join(save_path, f"model_epoch_{epoch+1}.pt")
        torch.save(unwrapped_model.state_dict(), model_save_path)
        print(f"Model saved to {model_save_path}")

    losses = np.array(losses)
    np.save(f'{save_path}/losses.npy', losses)

    # Generate a sample using the final model
    unwrapped_model = accelerator.unwrap_model(model)
    model_sample = sample(unwrapped_model, noise_scheduler, f'{save_path}/final_sample.png')


if __name__ == "__main__":
    args = parse_args()
    main(args)
