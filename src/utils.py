import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
import os
import random
from tqdm import tqdm
from torchvision import transforms
from PIL import Image


def get_constants(dataset_name='IllustrisTNG'):
    path = f'/groups/mlprojects/dm_diffusion/data/{dataset_name}_constants.pt'
    return torch.load(path)

def denormalize(tensor, mean, std): #TODO: make a lookup table, these are params for illustris
  """
  Denormalize a tensor image using mean and std.

  Args:
      tensor (torch.Tensor): The normalized tensor to be denormalized.
      mean (tuple): The mean used for normalization.
      std (tuple): The standard deviation used for normalization.

  Returns:
      torch.Tensor: The denormalized tensor.
  """
  return tensor * std + mean


def save_images_conditional(model, noise_scheduler, out_path, loader, device='cuda', verbose=False):
    model.eval()

    all_samples = []

    random_indicies = random.sample(range(0, len(loader.dataset) - 1), 100)
    gt = loader.dataset[random_indicies][0].to(device).permute(1, 0, 2, 3) 
    torch.save(gt, os.path.join(out_path, 'gt.pt'))
    guide = loader.dataset[random_indicies][1].to(device).permute(1, 0, 2, 3) 
    torch.save(guide, os.path.join(out_path, 'guide.pt'))
    
    for i in range(25):
        if verbose:
            print(f'{i}/25')
        sample = torch.randn(4, 1, model.unet.sample_size, model.unet.sample_size).to(device)

        curr_guide = loader.dataset[random_indicies[i*4:(i+1)*4]][1].to(device).permute(1, 0, 2, 3) 
        # Diffusion process
        for t in noise_scheduler.timesteps:
            with torch.no_grad():
                residual = model(sample, t, curr_guide)
            sample = noise_scheduler.step(residual, t, sample).prev_sample

        all_samples.append(sample.cpu())
    
    tog = torch.cat(all_samples)
    torch.save(tog, os.path.join(out_path, f'sample.pt'))

def save_images_diverse(model, noise_scheduler, out_path, loader, device='cuda'):
    model.eval()

    all_samples = []

    random_index= random.choice(range(0, len(loader.dataset) - 1))
    gt = loader.dataset[random_index][0].to(device).unsqueeze(1)
    torch.save(gt, os.path.join(out_path, 'gt.pt'))
    guide = loader.dataset[random_index][1].to(device).unsqueeze(1)
    torch.save(guide, os.path.join(out_path, 'guide.pt'))
    
    for i in range(100):
        sample = torch.randn(1, 1, model.unet.sample_size, model.unet.sample_size).to(device)
        # Diffusion process
        for t in noise_scheduler.timesteps:
            with torch.no_grad():
                residual = model(sample, t, guide)
            sample = noise_scheduler.step(residual, t, sample).prev_sample
        all_samples.append(sample.cpu())
    
    tog = torch.cat(all_samples)
    torch.save(tog, os.path.join(out_path, f'sample.pt'))


def save_loss_plot(file_path, save_path="loss_plot.png"):
    """
    Loads loss data from a .npy file and saves a plot of the losses as an image file.
    
    Args:
        file_path (str): Path to the .npy file containing loss data.
        save_path (str): Path to save the plot image file. Defaults to 'loss_plot.png'.
    """
    # Load the losses from the .npy file
    losses = np.load(file_path)

    # Plot the losses
    plt.figure(figsize=(10, 5))
    plt.plot(losses, label="Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss over Epochs")
    plt.legend()
    plt.grid(True)

    # Save the plot as an image file
    plt.savefig(save_path)
    plt.close()
    print(f"Loss plot saved to {save_path}")


def add_gaussian_noise(image, sigma=0.5):
    """
    Adds Gaussian noise to a PyTorch tensor image.
    """
    noise = torch.normal(mean=0.0, std=sigma, size=image.shape, device=image.device)
    noisy_image = image + noise
    return noisy_image

def blackout_pixels(img, percentage=10):
    """
    Black out 90% of the pixels in an image, retaining only the specified percentage.
    Handles both PIL images and tensors, and returns the same format as the input.

    Args:
        img (PIL.Image or torch.Tensor): Input image.
        percentage (int): Percentage of pixels to retain.

    Returns:
        PIL.Image or torch.Tensor: Transformed image with most pixels blacked out.
    """
    if isinstance(img, Image.Image):  # If input is a PIL image
        img_array = np.array(img)
        h, w, c = img_array.shape

        # Create mask
        mask = np.zeros((h, w), dtype=bool)
        num_pixels_to_keep = int((percentage / 100) * h * w)
        indices = np.random.choice(h * w, num_pixels_to_keep, replace=False)
        mask[np.unravel_index(indices, (h, w))] = True

        # Apply mask
        blacked_out_array = np.zeros_like(img_array)
        blacked_out_array[mask] = img_array[mask]

        return Image.fromarray(blacked_out_array)

    elif isinstance(img, torch.Tensor):  # If input is a tensor
        if img.dim() == 3:  # Expecting (C, H, W) format
            c, h, w = img.shape

            # Create mask
            mask = torch.zeros((h, w), dtype=torch.bool)
            num_pixels_to_keep = int((percentage / 100) * h * w)
            indices = torch.randperm(h * w)[:num_pixels_to_keep]
            mask.view(-1)[indices] = True

            # Apply mask to all channels
            blacked_out_tensor = torch.zeros_like(img)
            for channel in range(c):
                blacked_out_tensor[channel][mask] = img[channel][mask]

            return blacked_out_tensor

        else:
            raise ValueError("Expected a 3D tensor with shape (C, H, W).")

    else:
        raise TypeError("Input must be a PIL.Image or torch.Tensor.")
    
def sample_all(model, noise_scheduler, out_path, loader, config, constants=None, conditional=True, device='gpu', idx=None):
    """
    Paired loader, Globally normalized
    """
    model.eval()
    sample = torch.randn(1, 1, model.unet.sample_size, model.unet.sample_size).to(device)
    store = []
    timesteps = []

    if idx==None: 
        idx = random.randint(0, len(loader.dataset) - 1)
    gt = loader.dataset[idx][0].unsqueeze(0).to(device)
    fig = plt.figure()
    if constants == None:
        constants=get_constants(config["dataset"])
    gt_denormalized = denormalize(gt, mean=constants['dm_mean'], std=constants['dm_std'])
    gt_img = plt.imshow(gt_denormalized[0][0].cpu().numpy(), cmap='viridis')
    fig.colorbar(gt_img)
    plt.savefig(os.path.join(out_path, "gt.png"))
    plt.close()
    torch.save(gt, os.path.join(out_path, f'gt.pt'))

    guide = None
    if conditional: 
        cond_idx = 1
        guides = []
        if config["stellar"]:
            guide = add_gaussian_noise(loader.dataset[idx][cond_idx].unsqueeze(0).to(device), sigma=config['sigma_noise_stellar'])
            torch.save(guide, os.path.join(out_path, f'guide_stellar.pt'))
            fig = plt.figure()
            guide_denormalized = denormalize(guide, mean=constants['stellar_mean'], std=constants['stellar_std'])  # Denormalize the guide
            guide_img = plt.imshow(guide_denormalized[0][0].cpu().numpy(), cmap='viridis')
            fig.colorbar(guide_img)
            plt.savefig(os.path.join(out_path, "guide_stellar.png"))
            cond_idx += 1
            guides.append(guide)
        if config["frb"]:
            guide = loader.dataset[idx][cond_idx].unsqueeze(0).to(device)
            torch.save(guide, os.path.join(out_path, f'guide_frb.pt'))
            fig = plt.figure()
            guide_denormalized = denormalize(guide, mean=constants['gas_mean'], std=config['sigma_noise_stellar'])  # Denormalize the guide
            guide_img = plt.imshow(guide_denormalized[0][0].cpu().numpy(), cmap='viridis')
            fig.colorbar(guide_img)
            plt.savefig(os.path.join(out_path, "guide_frb.png"))
            cond_idx += 1
            guides.append(guide)
        if config["lensing"]:
            # lens1
            guide = add_gaussian_noise(loader.dataset[idx][cond_idx].unsqueeze(0).to(device), sigma=config["sigma_noise_lensing"])
            torch.save(guide, os.path.join(out_path, f'guide_lensing1.pt'))
            fig = plt.figure()
            guide_denormalized = denormalize(guide, mean=constants['lens1_mean'], std=constants['lens1_std'])  # Denormalize the guide
            guide_img = plt.imshow(guide_denormalized[0][0].cpu().numpy(), cmap='viridis')
            fig.colorbar(guide_img)
            plt.savefig(os.path.join(out_path, "guide_lens1.png"))
            cond_idx += 1
            guides.append(guide)
            #lens2
            guide = add_gaussian_noise(loader.dataset[idx][cond_idx].unsqueeze(0).to(device), sigma=config["sigma_noise_lensing"])
            torch.save(guide, os.path.join(out_path, f'guide_lensing2.pt'))
            fig = plt.figure()
            guide_denormalized = denormalize(guide, mean=constants['lens2_mean'], std=constants['lens2_std'])  # Denormalize the guide
            guide_img = plt.imshow(guide_denormalized[0][0].cpu().numpy(), cmap='viridis')
            fig.colorbar(guide_img)
            plt.savefig(os.path.join(out_path, "guide_lens2.png"))
            cond_idx += 1
            guides.append(guide)
        
        # add more for gravitational lensing

    for i, t in enumerate(noise_scheduler.timesteps):
        with torch.no_grad():
            residual = model(sample, t, guides)
        sample = noise_scheduler.step(residual, t, sample).prev_sample
        # Store the sample and its timestep label at 10 evenly spaced points
        if i == 0 or (i + 1) % (noise_scheduler.config.num_train_timesteps//10) == 0:
            store.append(sample[0].cpu())  # Store on CPU for plotting
            timesteps.append(t.item())  # Save the timestep number
    
    torch.save(sample, os.path.join(out_path, f'sample{i}.pt'))
    
    fig = plt.figure()
    final_img = store[-1][0].cpu().numpy()
    final_img_denormalized = denormalize(final_img, mean=constants['dm_mean'], std=constants['dm_std'])
    final_img_plot = plt.imshow(final_img_denormalized, cmap='viridis')
    fig.colorbar(final_img_plot)
    plt.savefig(os.path.join(out_path, "final_sample.png"))
    plt.close(fig)

    # Plot the images with their timestep labels and add a color bar
    f, axarr = plt.subplots(1, len(store), figsize=(30, 5), gridspec_kw={'wspace': 0.05})
    
    if len(store) == 1:
        axarr = [axarr]
    
    for i, (img, ax, timestep) in enumerate(zip(store, axarr, timesteps)):
        image = img[0].cpu().numpy()
        img_plot = ax.imshow(image, vmin=np.min(image), vmax=np.max(image), cmap="viridis")
        ax.axis("off")
        ax.set_title(f"Timestep {timestep}", fontsize=10)
        
        # Add individual color bar for each subplot
        cbar = f.colorbar(img_plot, ax=ax, orientation="vertical")
        cbar.ax.tick_params(labelsize=8)

    plt.savefig(os.path.join(out_path, "sample.png"), format='png')
    plt.close(f)
    print(f'Saved at {out_path}!')
    
    
def psnr_mse(sample, ground_truth):
    # sample and ground truth have shape (B, C, H, W) where B is num samples, C is channels, etc.
    sample = sample.cpu().float()
    ground_truth = ground_truth.cpu().float()
    mse = torch.mean((ground_truth - sample) ** 2)
    if mse==0: return float('inf')
    
    max_val = max(sample.max().item(), sample.max().item())
    return 20* torch.log10(max_val/torch.sqrt(mse)), mse


def plot_individual_img(img, save_path, mean, std, idx, title):
    img_denormalized = denormalize(img.cpu(), mean=mean, std=std) # Already denormalizes
    plt.figure(figsize=(8, 6))
    plt.imshow(img_denormalized[0][0], cmap='viridis')  # Choose a colormap (e.g., 'viridis', 'plasma', 'gray')
    plt.colorbar()  # Add colorbar with label
    plt.title(f"{title}, idx: {idx}")
    plt.savefig(save_path)
    plt.close()

def power(x,x2=None):
    signal_ndim = x.dim() - 2
    signal_size = x.shape[-signal_ndim:]
    
    kmax = min(s for s in signal_size) // 2
    even = x.shape[-1] % 2 == 0
    
    x = torch.fft.rfftn(x, s=signal_size)  # new version broke BC
    if x2 is None:
        x2 = x
    else:
        x2 = torch.fft.rfftn(x2, s=signal_size)
    P = x * x2.conj()
    
    P = P.mean(dim=0)
    P = P.sum(dim=0)
    
    del x, x2
    
    k = [torch.arange(d, dtype=torch.float32, device=P.device)
         for d in P.shape]
    k = [j - len(j) * (j > len(j) // 2) for j in k[:-1]] + [k[-1]]
    k = torch.meshgrid(*k,indexing="ij")
    k = torch.stack(k, dim=0)
    k = k.norm(p=2, dim=0)

    N = torch.full_like(P, 2, dtype=torch.int32)
    N[..., 0] = 1
    if even:
        N[..., -1] = 1

    k = k.flatten().real
    P = P.flatten().real
    N = N.flatten().real

    kbin = k.ceil().to(torch.int32)
    k = torch.bincount(kbin, weights=k * N)
    P = torch.bincount(kbin, weights=P * N)
    N = torch.bincount(kbin, weights=N).round().to(torch.int32)
    del kbin

    # drop k=0 mode and cut at kmax (smallest Nyquist)
    k = k[1:1+kmax]
    P = P[1:1+kmax]
    N = N[1:1+kmax]

    k /= N
    P /= N

    return k, P, N

#def compute_pk(field, field_b=None, boxsize=25):
def compute_pk(field, field_b=None, boxsize=6.25):
    k_conversion = 2*np.pi/boxsize
    # Assumes field has shape (1,1,Npixels, Npixels)
    assert len(field.shape) == 4
    if field_b is not None:
        assert len(field_b.shape) == 4
        k, pk, _ = power(
            torch.Tensor(field/np.sum(field)),
            torch.Tensor(field_b/np.sum(field_b)),
        )
    else:
        k, pk, _ = power(
            torch.Tensor(field/np.sum(field)),
        )
    k *= k_conversion
    pk *= boxsize**2
    return k, pk

def samplesP(samples):
  P_samples = []
  for img in samples:
    k, P = compute_pk((10**img).cpu().numpy())
    P_samples.append(P)
  P_stack = torch.stack(P_samples)
  mean_values = P_stack.mean(axis=0).cpu()
  return k, mean_values, P_samples


def power_spectra(all_samples, gt, out_path, dataset):
   
  k_gt, P_gt = compute_pk((10**gt).cpu().numpy())
  k_sample, sample_mean, P_sample = samplesP(all_samples)

  plt.figure(figsize=(12, 6))
  for p in P_sample:
    plt.loglog(k_sample.cpu(), p.cpu(), alpha = 0.1, color = 'orange')
  plt.loglog(k_sample.cpu(), sample_mean, label="Mean (Conditional Samples)", color = 'orange')
  plt.loglog(k_gt.cpu(), P_gt, label = "Mean (GT Maps)", color = 'blue')
  plt.xlabel("k")
  plt.ylabel("P")
  plt.title(dataset)
  plt.legend()
  plt.savefig(os.path.join(out_path, 'power_spectra.png'))

def corr_R(all_samples, gt, out_path, dataset):
  bs = 25
  R_samples = []
  for sample in all_samples:
    g = (10**gt).cpu().numpy()
    s = (10**sample).cpu().numpy()
    k = compute_pk(g, field_b=s, boxsize=bs)[0]
    corr = compute_pk(g, field_b=s, boxsize=bs)[1]/(np.sqrt(compute_pk(g, boxsize=bs)[1]) * np.sqrt(compute_pk(s, boxsize=bs)[1]))
    R_samples.append(corr)
  R_samples = np.array(R_samples)

  plt.figure(figsize=(12, 6))
  plt.plot(k, np.mean(R_samples, axis=0), label = "mean(R(k))")
  plt.fill_between(k, np.percentile(R_samples, 10, axis=0), np.percentile(R_samples, 90, axis=0), color="blue", alpha=0.2, label="Spread (10th pct - 90 pct)")
  plt.xlabel("k")
  plt.ylabel("R(k)")
  plt.ylim(0, 1)
  plt.legend(loc = 'lower left')
  plt.title(dataset)
  plt.savefig(os.path.join(out_path, "corr.png"))

def sample_multiple(model, noise_scheduler, out_path, loader, config, constants=None, conditional=True, device='gpu', idx=None, N=10):
    """
    Generate multiple samples using the same GT and guides, calculate the mean and std, 
    and save results including a combined visualization.
    """
    model.eval()
    if idx is None: 
        idx = random.randint(0, len(loader.dataset) - 1)
    gt = loader.dataset[idx][0].unsqueeze(0).to(device)
    guides = []
    
    if constants is None:
        constants = get_constants(config["dataset"])
    gt_denormalized = denormalize(gt, mean=constants['dm_mean'], std=constants['dm_std'])
    # Plot GT
    plot_individual_img(gt, save_path=os.path.join(out_path, "ground_truth.png"), \
        mean=constants['dm_mean'], std=constants['dm_std'], idx=idx, title='DM Guide')
    torch.save(gt_denormalized, os.path.join(out_path, "ground_truth.pt"))

    # Handle guide(s) based on configuration
    if conditional and config: 
        cond_idx = 1
        if config["stellar"]:
            stellar_guide = add_gaussian_noise(loader.dataset[idx][cond_idx].unsqueeze(0).to(device), sigma=config["sigma_noise_stellar"])
            guides.append(stellar_guide)
            torch.save(stellar_guide, os.path.join(out_path, "stellar_guide.pt"))
            cond_idx += 1
            # Plot Stellar
            plot_individual_img(stellar_guide, save_path=os.path.join(out_path, "stellar_guide.png"), \
                mean=constants['stellar_mean'], std=constants['stellar_std'], idx=idx, title='Stellar Guide')

        if config["frb"]:
            frb_guide =  loader.dataset[idx][cond_idx].unsqueeze(0).to(device) # already selects pixels differently based on iteration
            guides.append(frb_guide)
            torch.save(frb_guide, os.path.join(out_path, "frb_guide.pt"))
            cond_idx += 1
            # Plot FRB
            plot_individual_img(frb_guide, save_path=os.path.join(out_path, "frb_guide.png"), \
                mean=constants['gas_mean'], std=constants['gas_std'], idx=idx, title='FRB Guide')

        if config["lensing"]:
            lens1_guide =  add_gaussian_noise(loader.dataset[idx][cond_idx].unsqueeze(0).to(device), sigma=config["sigma_noise_lensing"]) # already selects pixels differently based on iteration
            guides.append(lens1_guide)
            torch.save(lens1_guide, os.path.join(out_path, "lens1_guide.pt"))
            cond_idx += 1
            lens2_guide =  add_gaussian_noise(loader.dataset[idx][cond_idx].unsqueeze(0).to(device), sigma=config["sigma_noise_lensing"]) # already selects pixels differently based on iteration
            guides.append(lens2_guide)
            torch.save(lens2_guide, os.path.join(out_path, "lens2_guide.pt"))
            cond_idx += 1
            # Plot lensing
            plot_individual_img(lens1_guide, save_path=os.path.join(out_path, "lens1_guide.png"), \
                mean=constants['lens1_mean'], std=constants['lens1_std'], idx=idx, title='Lens1 Guide')
            plot_individual_img(lens2_guide, save_path=os.path.join(out_path, "lens2_guide.png"), \
                mean=constants['lens2_mean'], std=constants['lens2_std'], idx=idx, title='Lens2 Guide')
    
    all_samples = []
    for run in range(N): # hard coded at N = 10 samples for now
        sample = torch.randn(1, 1, model.unet.sample_size, model.unet.sample_size).to(device)
        for t in noise_scheduler.timesteps:
            with torch.no_grad():
                residual = model(sample, t, guides)
            sample = noise_scheduler.step(residual, t, sample).prev_sample
        # Denormalize samples
        sample = denormalize(sample, mean=constants['dm_mean'], std=constants['dm_std'])
        all_samples.append(sample.cpu())
        torch.save(sample, os.path.join(out_path, f'sample_{run}.pt'))
    
    # Calculate mean and std
    stacked_samples = torch.stack(all_samples)  # Shape: (N, 1, H, W)
    mean_sample = stacked_samples.mean(dim=0)
    std_sample = stacked_samples.std(dim=0)

    # Calculate psnr
    gt_denormalized_dup = gt_denormalized.repeat(N, 1, 1, 1)
    psnr_samples, mse_samples = psnr_mse(stacked_samples, gt_denormalized_dup)
    print(f'Sample PSNR: {psnr_samples}, Sample MSE: {mse_samples}')
    torch.save({'psnr': psnr_samples, 'mse': mse_samples}, os.path.join(out_path, 'metrics.pt'))

    # Calculate correlations
    correlation_samples = calculate_correlation_exponentiated(gt_denormalized_dup, stacked_samples)
    plt.figure(figsize=(12, 6))
    plt.boxplot(
        [correlation_samples],
        patch_artist=True,
        labels=["DM vs Generated"],
        boxprops=dict(facecolor="skyblue"),
    )
    plt.ylabel("Correlation Coefficient")
    plt.title("Correlation Coefficients (Exponentiated)")
    corr_path = os.path.join(out_path, "dm_correlation.png")
    plt.savefig(corr_path)
    print(f"Boxplot saved to {corr_path}")

    # Save mean and std images
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    ax[0].imshow(gt_denormalized[0][0].cpu().numpy(), cmap='viridis')
    ax[0].set_title('Ground Truth')
    ax[0].axis('off')
    
    ax[1].imshow(mean_sample[0][0].numpy(), cmap='viridis')
    ax[1].set_title('Mean of Samples')
    ax[1].axis('off')
    
    ax[2].imshow(std_sample[0][0].numpy(), cmap='viridis')
    ax[2].set_title('Std of Samples')
    ax[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_path, "mean_std.png"))
    plt.close(fig)

    # Save all samples in a grid
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()
    for i, (sample, ax) in enumerate(zip(all_samples, axes)):
        im = ax.imshow(sample[0][0].numpy(), cmap='viridis')
        ax.set_title(f'Sample {i+1}')
        ax.axis('off')
        cbar = fig.colorbar(im, ax=ax, orientation='vertical', shrink=0.8)
        cbar.ax.tick_params(labelsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_path, "all_samples.png"))
    plt.close(fig)

    power_spectra(all_samples, gt_denormalized, out_path, config["dataset"])
    corr_R(all_samples, gt_denormalized, out_path, config["dataset"])

    print(f'All results saved at {out_path}!')
    return mean_sample, std_sample

def calculate_correlation_exponentiated(map1, map2):
    """Calculates correlations between two maps using exponentiation."""
    correlation_results = []
    for i in range(map1.shape[0]):
        map1_exp = np.exp(map1[i].squeeze().detach().cpu().numpy().flatten()) - 1e-8
        map2_exp = np.exp(map2[i].squeeze().detach().cpu().numpy().flatten()) - 1e-8
        corr = np.corrcoef(map1_exp, map2_exp)[0, 1]
        correlation_results.append(corr)
    return correlation_results


def power_spectra_comp(sample, gt):
  k_gt, P_gt = compute_pk((10**gt).cpu().numpy())
  k_sample, sample_mean, P_sample = samplesP(sample)
  return k_gt, P_gt, k_sample, sample_mean, P_sample

def corr_R_comp(all_samples, gt):
    bs = 25
    R_samples = []
    for sample in all_samples:
        g = (10**gt).cpu().numpy()
        s = (10**sample).cpu().numpy()
        corr = compute_pk(g, field_b=s, boxsize=bs)[1]/(np.sqrt(compute_pk(g, boxsize=bs)[1]) * np.sqrt(compute_pk(s, boxsize=bs)[1]))
        R_samples.append(corr)
    R_samples = np.array(R_samples)
    return R_samples

def sample_comparison(model, noise_scheduler, out_path, plot, change, num, loader, config, idx, constants=None, conditional=True, device='gpu', N=10):
    """
    Generate multiple samples using the same GT and guides, calculate the mean and std, 
    and save results including a combined visualization.
    """
    model.eval()
    gt = loader.dataset[idx][0].unsqueeze(0).to(device)
    guides = []
    
    if constants is None:
        constants = get_constants(config["dataset"])
    gt_denormalized = denormalize(gt, mean=constants['dm_mean'], std=constants['dm_std'])
    
    if plot:
        plot_individual_img(gt, save_path=os.path.join(out_path, "ground_truth.png"), \
            mean=constants['dm_mean'], std=constants['dm_std'], idx=idx, title='DM Guide')
        torch.save(gt_denormalized, os.path.join(out_path, "ground_truth.pt"))

    # Handle guide(s) based on configuration
    if conditional and config: 
        cond_idx = 1
        if config["stellar"]:
            stellar_guide = add_gaussian_noise(loader.dataset[idx][cond_idx].unsqueeze(0).to(device), sigma=config["sigma_noise_stellar"])
            guides.append(stellar_guide)
            cond_idx += 1
            if plot or change == "stellar":
                torch.save(stellar_guide, os.path.join(out_path, f"stellar_guide{num}.pt"))
                plot_individual_img(stellar_guide, save_path=os.path.join(out_path, f"stellar_guide{num}.png"), \
                    mean=constants['stellar_mean'], std=constants['stellar_std'], idx=idx, title='Stellar Guide')

        if config["frb"]:
            frb_guide =  loader.dataset[idx][cond_idx].unsqueeze(0).to(device) # already selects pixels differently based on iteration
            guides.append(frb_guide)
            cond_idx += 1
            if plot or change == "frb":
                torch.save(frb_guide, os.path.join(out_path, f"frb_guide{num}.pt"))
                plot_individual_img(frb_guide, save_path=os.path.join(out_path, f"frb_guide{num}.png"), \
                    mean=constants['gas_mean'], std=constants['gas_std'], idx=idx, title='FRB Guide')

        if config["lensing"]:
            lens1_guide =  add_gaussian_noise(loader.dataset[idx][cond_idx].unsqueeze(0).to(device), sigma=config["sigma_noise_lensing"]) # already selects pixels differently based on iteration
            guides.append(lens1_guide)
            cond_idx += 1
            lens2_guide =  add_gaussian_noise(loader.dataset[idx][cond_idx].unsqueeze(0).to(device), sigma=config["sigma_noise_lensing"]) # already selects pixels differently based on iteration
            guides.append(lens2_guide)
            cond_idx += 1
            # Plot lensing
            if plot or change == "lens":
                torch.save(lens1_guide, os.path.join(out_path, f"lens1_guide{num}.pt"))
                torch.save(lens2_guide, os.path.join(out_path, f"lens2_guide{num}.pt"))
                plot_individual_img(lens1_guide, save_path=os.path.join(out_path, f"lens1_guide{num}.png"), \
                    mean=constants['lens1_mean'], std=constants['lens1_std'], idx=idx, title='Lens1 Guide')
                plot_individual_img(lens2_guide, save_path=os.path.join(out_path, f"lens2_guide{num}.png"), \
                    mean=constants['lens2_mean'], std=constants['lens2_std'], idx=idx, title='Lens2 Guide')
    
    all_samples = []
    for run in range(N): # hard coded at N = 10 samples for now
        sample = torch.randn(1, 1, model.unet.sample_size, model.unet.sample_size).to(device)
        for t in noise_scheduler.timesteps:
            with torch.no_grad():
                residual = model(sample, t, guides)
            sample = noise_scheduler.step(residual, t, sample).prev_sample
        # Denormalize samples
        sample = denormalize(sample, mean=constants['dm_mean'], std=constants['dm_std'])
        all_samples.append(sample.cpu())
        torch.save(sample, os.path.join(out_path, f'sample_{num}_{run}.pt'))
    
     # Calculate mean and std
    stacked_samples = torch.stack(all_samples)  # Shape: (N, 1, H, W)
    mean_sample = stacked_samples.mean(dim=0)
    std_sample = stacked_samples.std(dim=0)

    # Calculate psnr
    gt_denormalized_dup = gt_denormalized.repeat(N, 1, 1, 1)
    psnr_samples, mse_samples = psnr_mse(stacked_samples, gt_denormalized_dup)
    print(f'Sample PSNR: {psnr_samples}, Sample MSE: {mse_samples}')
    torch.save({'psnr': psnr_samples, 'mse': mse_samples}, os.path.join(out_path, f'metrics{num}.pt'))

    # Calculate correlations
    correlation_samples = calculate_correlation_exponentiated(gt_denormalized_dup, stacked_samples)

    # Save mean and std images
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    ax[0].imshow(gt_denormalized[0][0].cpu().numpy(), cmap='viridis')
    ax[0].set_title('Ground Truth')
    ax[0].axis('off')
    
    ax[1].imshow(mean_sample[0][0].numpy(), cmap='viridis')
    ax[1].set_title('Mean of Samples')
    ax[1].axis('off')
    
    ax[2].imshow(std_sample[0][0].numpy(), cmap='viridis')
    ax[2].set_title('Std of Samples')
    ax[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_path, f"mean_std{num}.png"))
    plt.close(fig)

    # Save all samples in a grid
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()
    for i, (sample, ax) in enumerate(zip(all_samples, axes)):
        im = ax.imshow(sample[0][0].numpy(), cmap='viridis')
        ax.set_title(f'Sample {i+1}')
        ax.axis('off')
        cbar = fig.colorbar(im, ax=ax, orientation='vertical', shrink=0.8)
        cbar.ax.tick_params(labelsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_path, f"all_samples{num}.png"))
    plt.close(fig)

    return power_spectra_comp(all_samples, gt_denormalized), corr_R_comp(all_samples, gt_denormalized), correlation_samples
