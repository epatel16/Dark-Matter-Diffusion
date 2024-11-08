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


class DarkMatterUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_channels=64):
        super(DarkMatterUNet, self).__init__()
        
        # timestep embedding
        self.time_embed_dim = base_channels * 8 
        self.time_mlp = nn.Sequential(
            nn.Linear(self.time_embed_dim, self.time_embed_dim),
            nn.SiLU(),
            nn.Linear(self.time_embed_dim, self.time_embed_dim)
        )
        
        # encode
        self.enc1 = self.resnet_block(in_channels, base_channels)
        self.enc2 = self.resnet_block(base_channels, base_channels * 2)
        self.enc3 = self.resnet_block(base_channels * 2, base_channels * 4)  
        self.enc4 = self.resnet_block(base_channels * 4, base_channels * 8) 

        # bottleneck
        self.bottleneck = self.attn_conv_block(base_channels * 8, base_channels * 16)  

        # decode (updated w/ attention)
        self.upconv4 = nn.ConvTranspose2d(base_channels * 16, base_channels * 8, kernel_size=2, stride=2)
        self.dec4 = self.resnet_block(base_channels * 16, base_channels * 8)
        
        self.upconv3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.dec3 = self.resnet_block(base_channels * 8, base_channels * 4)
        
        self.upconv2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec2 = self.resnet_block(base_channels * 4, base_channels * 2)
        
        self.upconv1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec1 = self.resnet_block(base_channels * 2, base_channels)

        # out
        self.out_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def resnet_block(self, in_channels, out_channels):
        return ResNetBlock(in_channels, out_channels)

    def attn_conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(inplace=True),
            AttentionLayer(out_channels)
        )

    def forward(self, x, t=None):
        # ensure t is a tensor with batch dimension and integer type
        if t is not None:
            if isinstance(t, int) or isinstance(t, float):
                t = torch.tensor([t], dtype=torch.int64, device=x.device)
            elif t.dim() == 0:
                t = t.unsqueeze(0).to(dtype=torch.int64)

            # sinusoidal embedding
            t_emb = self.sinusoidal_embedding(t, self.time_embed_dim)
            t_emb = self.time_mlp(t_emb).view(x.shape[0], -1, 1, 1)  # Output shape: [batch_size, time_embed_dim, 1, 1]
        else:
            t_emb = torch.zeros((x.shape[0], self.time_embed_dim, 1, 1), device=x.device)
        
        # encode
        enc1 = self.enc1(x)
        enc2 = self.enc2(F.max_pool2d(enc1, kernel_size=2))
        enc3 = self.enc3(F.max_pool2d(enc2, kernel_size=2))
        enc4 = self.enc4(F.max_pool2d(enc3, kernel_size=2))

        # timestep embedding (diff)
        enc4 = enc4 + t_emb.expand(-1, enc4.shape[1], enc4.shape[2], enc4.shape[3])  # Broadcasting timestep embedding correctly
        
        # middle
        bottleneck = self.bottleneck(F.max_pool2d(enc4, kernel_size=2))

        # decode
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((enc4, dec4), dim=1)
        dec4 = self.dec4(dec4)
        
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((enc3, dec3), dim=1)
        dec3 = self.dec3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((enc2, dec2), dim=1)
        dec2 = self.dec2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((enc1, dec1), dim=1)
        dec1 = self.dec1(dec1)

        # Output layer
        out = self.out_conv(dec1)
        return out

    def sinusoidal_embedding(self, timesteps, dim):
        if timesteps.dim() == 0:
            timesteps = timesteps.unsqueeze(0)

        half_dim = dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)
        
        # If the input and output channels are different, add a skip connection
        if in_channels != out_channels:
            self.skip_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip_conv = None

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        
        if self.skip_conv is not None:
            identity = self.skip_conv(identity)
        
        out += identity
        out = self.relu(out)
        return out

class AttentionLayer(nn.Module):
    def __init__(self, channels):
        super(AttentionLayer, self).__init__()
        self.query_conv = nn.Conv2d(channels, channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(channels, channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, C, H, W = x.size()
        query = self.query_conv(x).view(batch_size, -1, H * W).permute(0, 2, 1)  # B x N x C//8
        key = self.key_conv(x).view(batch_size, -1, H * W)  # B x C//8 x N
        attention = torch.bmm(query, key)  # B x N x N
        attention = self.softmax(attention)
        value = self.value_conv(x).view(batch_size, -1, H * W)  # B x C x N

        out = torch.bmm(value, attention.permute(0, 2, 1))  # B x C x N
        out = out.view(batch_size, C, H, W)

        return out + x  # residual connection
