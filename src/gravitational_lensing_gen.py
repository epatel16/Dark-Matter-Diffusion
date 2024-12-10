import numpy as np
import torch
import torch.nn as nn
from datasets import SlicedDataset
from torchvision import transforms
import argparse
import os

def mass_to_shear(mass):
    """
    Uses the mass as an estimate of the convergence of the lens (kappa),
    and then performs Kaiser & Squires (1993) to convert convergence to shear.
    """
    kappa = (mass - mass.mean()) / mass.mean()
    ny, nx = kappa.shape
    sy, sx = ny * 2 - 1, nx * 2 - 1
    kappa_fft = torch.fft.fft2(torch.tensor(kappa), s=(sy, sx))
    fx, fy = torch.meshgrid(torch.fft.fftfreq(sy), torch.fft.fftfreq(sx), indexing="ij")
    denom = fx**2 + fy**2
    denom[0, 0] = torch.inf  # Avoid division by zero
    kernel_1 = ((fy**2 - fx**2) / denom).to(device="cuda")
    kernel_2 = (2 * fx * fy / denom).to(device="cuda")
    g1 = torch.fft.ifft2(kappa_fft * kernel_1).real
    g2 = torch.fft.ifft2(kappa_fft * kernel_2).real
    return g1[:ny, :nx], g2[:ny, :nx]

def generate_shear_maps(dataloader):
    """
    Processes mass maps in chunks to avoid memory issues.
    """
    g1s = []
    g2s = []

    for chunk in dataloader:
        for mass in chunk:
            g1, g2 = mass_to_shear(mass.to(device="cuda"))
            g1s.append(g1.cpu().unsqueeze(0))
            g2s.append(g2.cpu().unsqueeze(0))
    return torch.cat(g1s, dim=0), torch.cat(g2s, dim=0)

def parse_args():
    parser = argparse.ArgumentParser(description="A script to parse training parameters.")
    
    # Adding arguments
    parser.add_argument("--dataset", default="IllustrisTNG", type=str, required=False, help="IllustrisTNG, Astrid, or SIMBA.")

    # Parsing arguments
    return parser.parse_args()

def grav_lensing(args):
    
    torch.cuda.empty_cache()

    npy_prefix = "/groups/mlprojects/dm_diffusion/data/"
    npy_file = f"Maps_Mtot_{args.dataset}_LH_z=0.00.npy"

    transform = transforms.Compose([
        transforms.Lambda(lambda x: ((x - x.min()) / (x.max() - x.min()) - 0.5) * 2)
    ])

    dm_dataset = SlicedDataset(os.path.join(npy_prefix, npy_file), transform=transform)
    g1_maps, g2_maps = generate_shear_maps(dm_dataset)
    torch.save(npy_prefix + f'{args.dataset}_g1_maps.pt', g1_maps.cuda())
    torch.save(npy_prefix + f'{args.dataset}_g2_maps.pt', g2_maps.cuda())

if __name__ == "__main__":
    args = parse_args()
    grav_lensing(args)