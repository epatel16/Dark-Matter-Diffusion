import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import UNet2DModel
import math
import matplotlib.pyplot as plt

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
        self.enc1 = self.conv_block(in_channels, base_channels)
        self.enc2 = self.conv_block(base_channels, base_channels * 2)
        self.enc3 = self.attn_conv_block(base_channels * 2, base_channels * 4)  
        self.enc4 = self.attn_conv_block(base_channels * 4, base_channels * 8) 

        # bottleneck
        self.bottleneck = self.attn_conv_block(base_channels * 8, base_channels * 16)  

        # decode (updated w/ attention)
        self.upconv4 = nn.ConvTranspose2d(base_channels * 16, base_channels * 8, kernel_size=2, stride=2)
        self.dec4 = self.attn_conv_block(base_channels * 16, base_channels * 8)  # Added attention here
        
        self.upconv3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.dec3 = self.attn_conv_block(base_channels * 8, base_channels * 4)  # Added attention here
        
        self.upconv2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(base_channels * 4, base_channels * 2)
        
        self.upconv1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(base_channels * 2, base_channels)

        # out
        self.out_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(inplace=True)
        )

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
