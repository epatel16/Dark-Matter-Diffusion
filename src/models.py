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
import math

from tqdm import tqdm
from datasets import SlicedDataset, NPYDataset
from accelerate import Accelerator
from torch.optim.lr_scheduler import CosineAnnealingLR

from datetime import datetime
import os
import argparse

class DiffUNet_MultiCond(nn.Module):
    def __init__(self, config, conditional=True):
        super(DiffUNet_MultiCond, self).__init__()
        self.unet = UNet2DModel(
            sample_size=config["image_size"],
            in_channels=config["conditioning_channels"] + config['target_channels'],
            out_channels=config['target_channels'],
            down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D", "AttnDownBlock2D"),
            up_block_types=("AttnUpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"),
            block_out_channels=(48, 96, 192, 384),
            layers_per_block=2,
            norm_num_groups=8,
            act_fn="silu",
            attention_head_dim=8,
            add_attention=True,
            downsample_type='resnet',
            upsample_type='resnet'
        )
        self.conditional = conditional
        self.config = config

    def forward(self, x, t, condition=None):
        for cond in condition:
            cond = cond.expand(-1, -1, x.size(2), x.size(3))
            x = torch.cat((x, cond), dim=1)
        return self.unet(x, t).sample
    

class DiffUNet(nn.Module):
    def __init__(self, config, conditional=True):
        super(DiffUNet, self).__init__()
        self.unet = UNet2DModel(
            sample_size=config["image_size"],
            in_channels=config["conditioning_channels"] + config['target_channels'],
            out_channels=config['target_channels'],
            down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D", "AttnDownBlock2D"),
            up_block_types=("AttnUpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"),
            block_out_channels=(48, 96, 192, 384),
            layers_per_block=2,
            norm_num_groups=8,
            act_fn="silu",
            attention_head_dim=8,
            add_attention=True,
            downsample_type='resnet',
            upsample_type='resnet'
        )
        self.conditional = conditional

    def forward(self, x, t, condition=None):
        if self.conditional:
            condition = condition.expand(-1, -1, x.size(2), x.size(3))
            x = torch.cat((x, condition), dim=1)
        return self.unet(x, t).sample

class GravUNet(nn.Module):
    def __init__(self, config, conditional=True):
        super(GravUNet, self).__init__()
        self.unet = UNet2DModel(
            sample_size=config["image_size"],
            in_channels=config["conditioning_channels"] + config['target_channels'],
            out_channels=config['target_channels'],
            down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D", "AttnDownBlock2D"),
            up_block_types=("AttnUpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"),
            block_out_channels=(48, 96, 192, 384),
            layers_per_block=2,
            norm_num_groups=8,
            act_fn="silu",
            attention_head_dim=8,
            add_attention=True,
            downsample_type='resnet',
            upsample_type='resnet'
        )
        self.conditional = conditional
        self.config = config

    def forward(self, x, t, condition=None):
        x = torch.cat((x, condition), dim=1)
        return self.unet(x, t).sample
