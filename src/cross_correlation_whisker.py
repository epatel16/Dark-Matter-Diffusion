import os
import torch
import numpy as np
from matplotlib import pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
from datasets import SlicedDataset
from diffusers import DDPMScheduler
from models import DiffUNet
from datetime import datetime
from utils import get_constants, denormalize
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Generate samples and compute correlations.")
    parser.add_argument("--dataset", default="IllustrisTNG", type=str, help="Dataset name (IllustrisTNG, Astrid, SIMBA).")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model (.pt file).")
    parser.add_argument("--out_path", type=str, default="./output", help="Directory to save outputs.")
    parser.add_argument("--batch_size", type=int, default=12, help="Batch size for DataLoader.")
    parser.add_argument("--timesteps", type=int, default=1000, help="Number of timesteps for the diffusion process.")
    parser.add_argument("--img_size", type=int, default=64, help="Size of the generated images.")
    parser.add_argument("--num_samples_per_map", type=int, default=5, help="Number of samples to generate per map.")
    return parser.parse_args()


def generate_multiple_samples(model, conditioning_maps, noise_scheduler, num_samples_per_map, device, img_size):
    """Generates multiple samples using the diffusion model."""
    model.eval()
    all_samples = []
    with torch.no_grad():
        for _ in range(num_samples_per_map):
            x = torch.randn((conditioning_maps.size(0), 1, img_size, img_size)).to(device)
            for t in reversed(range(noise_scheduler.num_train_timesteps)):
                timesteps = torch.full((conditioning_maps.size(0),), t, device=device, dtype=torch.long)
                residual = model(x, timesteps, conditioning_maps)
                x = noise_scheduler.step(residual, t, x).prev_sample
            all_samples.append(x.cpu())
    return np.array(all_samples).transpose(1, 0, 2, 3, 4)  # [num_maps, num_samples, ...]


def calculate_correlation_exponentiated(map1, map2):
    """Calculates correlations between two maps using exponentiation."""
    correlation_results = []
    for i in range(map1.shape[0]):
        map1_exp = np.exp(map1[i].squeeze().detach().cpu().numpy().flatten()) - 1e-8
        map2_exp = np.exp(map2[i].squeeze().detach().cpu().numpy().flatten()) - 1e-8
        corr = np.corrcoef(map1_exp, map2_exp)[0, 1]
        correlation_results.append(corr)
    return correlation_results


def main():
    args = parse_args()

    # Configuration and constants
    config = {
        "batch_size": args.batch_size,
        "num_timesteps": args.timesteps,
        "image_size": args.img_size,
        "conditioning_channels": 1,
        "target_channels": 1,
        "stellar_file": f"/groups/mlprojects/dm_diffusion/data/Maps_Mstar_{args.dataset}_LH_z=0.00.npy",
        "dm_file": f"/groups/mlprojects/dm_diffusion/data/Maps_Mcdm_{args.dataset}_LH_z=0.00.npy",
        "num_workers": 1,
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define transformations
    transform = transforms.Compose([
        transforms.Lambda(lambda x: torch.log10(x + 1e-8)),  # Log transformation
        transforms.Resize((config["image_size"], config["image_size"])),  # Resize to 64x64
        transforms.Lambda(lambda x: ((x - x.min()) / (x.max() - x.min()) - 0.5) * 2)  # Normalize to [-1, 1]
    ])

    # Initialize datasets using memory-mapped SlicedDataset
    stellar_dataset = SlicedDataset(file_path=config["stellar_file"], transform=transform)
    dm_dataset = SlicedDataset(file_path=config["dm_file"], transform=transform)

    # constants = get_constants()

    # transform_dm = transforms.Compose([
    #     transforms.Lambda(lambda x: torch.log10(x + 1e-8)),  # Log transformation
    #     transforms.Resize((config["image_size"], config["image_size"])),  # Resize to 64x64
    #     transforms.Lambda(lambda x: (x - constants['dm_mean']) / constants['dm_std'])
    # ])
    
    # transform_stellar = transforms.Compose([
    #     transforms.Lambda(lambda x: torch.log10(x + 1e-8)),  # Log transformation
    #     transforms.Resize((config["image_size"], config["image_size"])),  # Resize to 64x64
    #     transforms.Lambda(lambda x: (x - constants['stellar_mean']) / constants['stellar_std'])
    # ])

    # stellar_dataset = SlicedDataset(file_path=config["stellar_file"], transform=transform_stellar)
    # dm_dataset = SlicedDataset(file_path=config["dm_file"], transform=transform_dm)

    # DataLoader for batching
    stellar_dataloader = DataLoader(stellar_dataset, batch_size=config["batch_size"], shuffle=False)
    dm_dataloader = DataLoader(dm_dataset, batch_size=config["batch_size"], shuffle=False)

    # Initialize the noise scheduler
    noise_scheduler = DDPMScheduler(num_train_timesteps=config["num_timesteps"])

    # Load model
    model = DiffUNet(config, conditional=True).to(device)
    model.load_state_dict(torch.load(args.model_path))
    model.eval()


    # Generate samples and compute correlations
    stellar_maps = next(iter(stellar_dataloader)).to(device)
    dm_maps = next(iter(dm_dataloader)).to(device)

    multiple_samples_stellar = generate_multiple_samples(model, stellar_maps, noise_scheduler, args.num_samples_per_map, device, config["image_size"])

    mean_generated_samples = torch.tensor(multiple_samples_stellar.mean(axis=1), device=device)

    correlation_stellar_generated = calculate_correlation_exponentiated(stellar_maps, mean_generated_samples)

    correlation_dm_generated = calculate_correlation_exponentiated(dm_maps, mean_generated_samples)

    # Save visualizations
    os.makedirs(args.out_path, exist_ok=True)
    plt.figure(figsize=(12, 6))
    plt.boxplot(
        [correlation_stellar_generated, correlation_dm_generated],
        patch_artist=True,
        labels=["Stellar vs Generated", "DM vs Generated"],
        boxprops=dict(facecolor="skyblue"),
    )
    plt.ylabel("Correlation Coefficient")
    plt.title("Correlation Coefficients (Exponentiated)")
    output_plot = os.path.join(args.out_path, "correlation.png")
    plt.savefig(output_plot)
    print(f"Boxplot saved to {output_plot}")


if __name__ == "__main__":
    main()

