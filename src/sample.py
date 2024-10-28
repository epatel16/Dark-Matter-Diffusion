import numpy as np
import torch
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
from utils import sample, LogTransform

from datetime import datetime
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="A script to parse training parameters.")
    
    # Adding arguments
    parser.add_argument("--model_path", type=str, required=True, help="Path to specific model.")
    parser.add_argument("--out_path", type=str, required=False, default=None, help="Path to save models.")
    parser.add_argument("--img_size", type=int, required=False, default=64, help="Image size. Single int, (H = W).")
    
    # Parsing arguments
    return parser.parse_args()


def main(args):
    # Assuming that the model path is specific enough
    ep = args.model_path.split('_')[-1].split('.')[0]
    save_path = '/'.join(args.model_path.split('/')[:-1])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet2DModel(
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
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    # model=torch.load(args.model_path,  map_location=torch.device('cpu'))
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
    model_sample = sample(model, noise_scheduler, f'{save_path}/ep{ep}_sample.png')


    
if __name__ == "__main__":
    args = parse_args()
    main(args)