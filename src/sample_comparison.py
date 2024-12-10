import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from datasets import PairedDataset

from diffusers import DDPMScheduler
from models import DiffUNet_MultiCond

from utils import sample_comparison, get_constants, blackout_pixels

from datetime import datetime
import os
import argparse
import yaml
import matplotlib.pyplot as plt
import numpy as np
import random 

def parse_args():
    parser = argparse.ArgumentParser(description="A script to parse training parameters.")
    
    # Adding arguments
    # parser.add_argument("--dataset", default="IllustrisTNG", type=str, required=False, help="IllustrisTNG, Astrid, or SIMBA.")
    parser.add_argument("--baseline_path", type=str, required=True, help="Path to baseline model.")
    parser.add_argument("--mod1_path", type=str, required=True, help="Path to mod1 model.")
    parser.add_argument("--mod2_path", type=str, required=True, help="Path to mod2 model.")
    parser.add_argument("--change", type=str, required=True, help="Either, stellar, frb, or lens.")
    parser.add_argument("--batch_size", type=int, required=False, default=12, help="Batch size for training.")
    parser.add_argument("--idx", type=int, required=False, default=None, help="Specific index in dataset of sample.")
    
    
    # Parsing arguments
    return parser.parse_args()


def main(args):

    k_gts = None
    P_gts = None
    k_samples = []
    sample_means = []
    P_samples = []
    R_samples = []
    correlation_samples = []
    
    paths = [args.baseline_path, args.mod1_path, args.mod2_path]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(f"/groups/mlprojects/dm_diffusion/power_spectra/Astrid/comp/{args.change}", f'sampling_{timestamp}')
    os.makedirs(out_path, exist_ok=True)

    plot = True

    idx = args.idx
    if idx is None:
        idx = random.randint(0, 14999)

    for i, path in enumerate(paths):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # e.g. /groups/mlprojects/dm_diffusion/model_out/lr0.0001_step1000_size64_condTrue/20241118_182815/model_epoch_10.pt
        config_path = os.path.dirname(path)
        
        with open(os.path.join(config_path, "config.yaml"), "r") as file:
            config = yaml.safe_load(file)
        constants = get_constants(config["dataset"])

        transform_dm = transforms.Compose([
            transforms.Lambda(lambda x: torch.log10(x + 1e-8)),  # Log transformation
            transforms.Resize((config["image_size"], config["image_size"])),  # Resize to 64x64
            transforms.Lambda(lambda x: (x - constants['dm_mean']) / constants['dm_std'])
        ])
        
        transform_stellar = transforms.Compose([
            transforms.Lambda(lambda x: torch.log10(x + 1e-8)),  # Log transformation
            transforms.Resize((config["image_size"], config["image_size"])),  # Resize to 64x64
            transforms.Lambda(lambda x: (x - constants['stellar_mean']) / constants['stellar_std'])
        ])

        transform_gas = transforms.Compose([
            transforms.Lambda(lambda x: torch.log10(x + 1e-8)),  # Log transformation
            transforms.Resize((config["image_size"], config["image_size"])),  # Resize to specified size
            transforms.Lambda(lambda x: (x - constants['gas_mean']) / constants['gas_std']),
            transforms.Lambda(lambda img: blackout_pixels(img, percentage=config['perc_preserved_frb'])),  # Blackout 90% of pixels
        ])
        
        transform_lens1 = transforms.Compose([
            transforms.Resize((config["image_size"], config["image_size"])),  # Resize to 64x64
            transforms.Lambda(lambda x: (x - constants['lens1_mean']) / constants['lens1_std']), # TODO: change to correct constants
        ])
        
        transform_lens2 = transforms.Compose([
            transforms.Resize((config["image_size"], config["image_size"])),  # Resize to 64x64
            transforms.Lambda(lambda x: (x - constants['lens2_mean']) / constants['lens2_std']), # TODO: change to correct constants
        ])

        filenames = [config["dm_file"]]
        transform = [transform_dm]
        if config["conditional"]:
            if config["stellar"]:
                filenames.append(config["stellar_file"])
                transform.append(transform_stellar)
            if config["frb"]:
                filenames.append(config["gas_file"])
                transform.append(transform_gas)
            if config["lensing"]:
                filenames.append(config["lensing1_file"])
                transform.append(transform_lens1)
                filenames.append(config["lensing2_file"])
                transform.append(transform_lens2)

        paired_dataset = PairedDataset(filenames, transform=transform)
        PairedDataloader = DataLoader(paired_dataset, batch_size=config["batch_size"], shuffle=True, num_workers = config["num_workers"])

        noise_scheduler = DDPMScheduler(num_train_timesteps=config["num_timesteps"], clip_sample=False)
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = DiffUNet_MultiCond(config, conditional=config['conditional']).to(device)

        model.to(device)
        model.load_state_dict(torch.load(path))

        (k_gt, P_gt, k_sample, sample_mean, P_sample), R_sample, correlation_sample = sample_comparison(model, noise_scheduler, out_path, plot, args.change, i, loader=PairedDataloader, config=config, idx=idx, constants=constants, conditional=config["conditional"], device=device)
        if plot:
            k_gts = k_gt
            P_gts = P_gt
        k_samples.append(k_sample)
        sample_means.append(sample_mean)
        P_samples.append(P_sample)
        R_samples.append(R_sample)
        correlation_samples.append(correlation_sample)
        
        plot = False

    descriptions = []
    if args.change == "frb":
        descriptions = ['frb_1', 'frb_5', 'frb_10']
    elif args.change == "lens":
        descriptions = ['lens_10', 'lens_5', 'lens_1']
    elif args.change == "stellar":
        descriptions = ['stellar_0.1', 'stellar_0.01', 'stellar_0']
        
    colors = ["blue", "orange", "green"]
    plt.figure(figsize=(12, 6))
    for k_sample, sample_mean, P_sample, desc, col in zip(k_samples, sample_means, P_samples, descriptions, colors):
        for p in P_sample:
            plt.loglog(k_sample.cpu(), p.cpu(), alpha = 0.1, color = col)
        plt.loglog(k_sample.cpu(), sample_mean, label= f"Mean ({desc} Samples)", color = col)
    plt.loglog(k_gts.cpu(), P_gts, label = "Mean (GT Maps)", color = 'black')
    plt.xlabel("k")
    plt.ylabel("P")
    plt.title(config['dataset'])
    plt.legend()
    plt.savefig(os.path.join(out_path, 'power_spectra_comp.png'))

    plt.figure(figsize=(12, 6))
    for k_sample, R_sample, desc, col in zip(k_samples, R_samples, descriptions, colors):
        plt.plot(k_sample, np.mean(R_sample, axis=0), label = f"{desc}mean(R(k))")
        plt.fill_between(k_sample, np.percentile(R_sample, 10, axis=0), np.percentile(R_sample, 90, axis=0), color=col, alpha=0.2, label=f"{desc} Spread (10th pct - 90 pct)")
    plt.xlabel("k")
    plt.ylabel("R(k)")
    plt.ylim(0, 1)
    plt.legend(loc = 'lower left')
    plt.title(config['dataset'])
    plt.savefig(os.path.join(out_path, "corr_comp.png"))

    plt.figure(figsize=(12, 6))
    plt.boxplot(
        correlation_samples,
        patch_artist=True,
        labels= [f"DM vs. {desc}" for desc in descriptions],
        boxprops=dict(facecolor="skyblue"),
    )
    plt.ylabel("Correlation Coefficient")
    plt.title("Correlation Coefficients (Exponentiated)")
    plt.savefig(os.path.join(out_path, f"dm_correlation.png"))

    print(f"sample succesfully saved in {out_path}")

    
if __name__ == "__main__":
    args = parse_args()
    main(args) 