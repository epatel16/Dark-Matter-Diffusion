import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
import os
import random
from tqdm import tqdm

# for globally normalized data
def sample_v2(model, noise_scheduler, out_path, condition_loader=None, device='cpu'):
    """
    Global Normalization
    """
    model.eval()
    sample = torch.randn(1, 1, model.unet.sample_size, model.unet.sample_size).to(device)
    store = []
    timesteps = []

    guide = None
    if condition_loader:
        random_index = random.randint(0, len(condition_loader.dataset) - 1)
        guide = condition_loader.dataset[random_index].unsqueeze(0).to(device)
        # guide = denormalize(guide, mean=0.11826974898576736, std=1.0741989612579346)  # Denormalize the guide
        torch.save(guide, os.path.join(out_path, 'guide.pt'))
        fig = plt.figure()
        guide_img = plt.imshow(guide[0][0].cpu().numpy(), cmap='viridis')
        fig.colorbar(guide_img)
        plt.savefig(os.path.join(out_path, "guide.png"))
        plt.close(fig)

    for i, t in enumerate(noise_scheduler.timesteps):
        with torch.no_grad():
            residual = model(sample, t, guide)
        sample = noise_scheduler.step(residual, t, sample).prev_sample
        
        # Store the sample and its timestep label at 10 evenly spaced points
        if i == 0 or (i + 1) % (noise_scheduler.config.num_train_timesteps // 10) == 0:
            store.append(sample[0].cpu())  # Store on CPU for plotting
            timesteps.append(t.item())  # Save the timestep number
            torch.save(sample, os.path.join(out_path, f'sample{i}.pt'))

    # Denormalize all stored samples
    # store = [denormalize(img.unsqueeze(0), mean=10.971004486083984, std=0.5090954303741455)[0] for img in store]
    
    fig = plt.figure()
    final_img = store[-1][0].cpu().numpy()
    final_img_plot = plt.imshow(final_img, vmin=np.min(final_img), vmax=np.max(final_img), cmap='viridis')
    # final_img_plot = plt.imshow(final_img, vmin=-1, vmax=4, cmap='viridis')
    print(f'min: {np.min(final_img)}, max: {np.max(final_img)}')
    fig.colorbar(final_img_plot)
    plt.savefig(os.path.join(out_path, "final_sample.png"))
    plt.close(fig)
    
    # Create a plot with individual subplots and color bars
    num_images = len(store)
    f, axarr = plt.subplots(1, num_images, figsize=(5 * len(store), 5), gridspec_kw={'wspace': 0.3})

    if num_images == 1:
        axarr = [axarr]

    for i, (img, ax, timestep) in enumerate(zip(store, axarr, timesteps)):
        image = img[0].cpu().numpy()
        img_plot = ax.imshow(image, vmin=np.min(image), vmax=np.max(image), cmap="viridis")
        # img_plot = ax.imshow(image, vmin=-1, vmax=4, cmap="viridis")
        ax.axis("off")
        ax.set_title(f"Timestep {timestep}", fontsize=10)
        
        # Add individual color bar for each subplot
        cbar = f.colorbar(img_plot, ax=ax, orientation="vertical")
        cbar.ax.tick_params(labelsize=8)

    plt.savefig(os.path.join(out_path, "sample.png"), format='png')
    plt.close(f)
    print(f'Saved at {out_path}!')


# for individually normalized data
def sample_v1(model, noise_scheduler, out_path, condition_loader=None, device='cpu'):
    """

    """
    model.eval()
    sample = torch.randn(1, 1, model.unet.sample_size, model.unet.sample_size).to(device)
    store = []
    timesteps = []

    guide = None
    if condition_loader:
        random_index = random.randint(0, len(condition_loader.dataset) - 1)
        guide = condition_loader.dataset[random_index].unsqueeze(0).to(device)
        torch.save(guide, os.path.join(out_path, f'guide.pt'))
        fig = plt.figure()
        guide_img = plt.imshow(guide[0][0].cpu().numpy(), cmap='viridis')
        fig.colorbar(guide_img)
        plt.savefig(os.path.join(out_path, "guide.png"))

    for i, t in enumerate(noise_scheduler.timesteps):
        with torch.no_grad():
            residual = model(sample, t, guide)
        sample = noise_scheduler.step(residual, t, sample).prev_sample
        # Store the sample and its timestep label at 10 evenly spaced points
        if i == 0 or (i + 1) % (noise_scheduler.config.num_train_timesteps//10) == 0:
            store.append(sample[0].cpu())  # Store on CPU for plotting
            timesteps.append(t.item())  # Save the timestep number
            torch.save(sample, os.path.join(out_path, f'sample{i}.pt'))

    # Determine min and max for consistent color scaling
    vmin = min(img.min().item() for img in store)
    vmax = max(img.max().item() for img in store)
    
    fig = plt.figure()
    final_img = store[-1][0].cpu().numpy()
    final_img_plot = plt.imshow(final_img, cmap='viridis')
    fig.colorbar(final_img_plot)
    plt.savefig(os.path.join(out_path, "final_sample.png"))
    plt.close(fig)

    # Plot the images with their timestep labels and add a color bar
    f, axarr = plt.subplots(1, len(store), figsize=(30, 5), gridspec_kw={'wspace': 0.05})
    
    if len(store) == 1:
        axarr = [axarr]
    
    for i, (img, ax, timestep) in enumerate(zip(store, axarr, timesteps)):
        image = img[0].cpu().numpy()
        img_plot = ax.imshow(image, cmap="viridis")
        ax.axis("off")
        ax.set_title(f"Timestep {timestep}", fontsize=10)
        
        # Add individual color bar for each subplot
        cbar = f.colorbar(img_plot, ax=ax, orientation="vertical")
        cbar.ax.tick_params(labelsize=8)

    plt.savefig(os.path.join(out_path, "sample.png"), format='png')
    plt.close(f)
    print(f'Saved at {out_path}!')

def sample_v3(model, noise_scheduler, out_path, loader, conditional=True, device='cpu'):
    """
    Individually normalized, paired loader
    """
    model.eval()
    sample = torch.randn(1, 1, model.unet.sample_size, model.unet.sample_size).to(device)
    store = []
    timesteps = []


    random_index = random.randint(0, len(loader.dataset) - 1)
    gt = loader.dataset[random_index][0].unsqueeze(0).to(device)
    fig = plt.figure()
    gt_img = plt.imshow(gt[0][0].cpu().numpy(), cmap='viridis')
    fig.colorbar(gt_img)
    plt.savefig(os.path.join(out_path, "gt.png"))
    torch.save(gt, os.path.join(out_path, f'gt.pt'))

    guide = None
    if conditional:
        guide = loader.dataset[random_index][1].unsqueeze(0).to(device)
        torch.save(guide, os.path.join(out_path, f'guide.pt'))
        fig = plt.figure()
        guide_img = plt.imshow(guide[0][0].cpu().numpy(), cmap='viridis')
        fig.colorbar(guide_img)
        plt.savefig(os.path.join(out_path, "guide.png"))

    for i, t in enumerate(noise_scheduler.timesteps):
        with torch.no_grad():
            residual = model(sample, t, guide)
        sample = noise_scheduler.step(residual, t, sample).prev_sample
        # Store the sample and its timestep label at 10 evenly spaced points
        if i == 0 or (i + 1) % (noise_scheduler.config.num_train_timesteps//10) == 0:
            store.append(sample[0].cpu())  # Store on CPU for plotting
            timesteps.append(t.item())  # Save the timestep number
            torch.save(sample, os.path.join(out_path, f'sample{i}.pt'))
    
    fig = plt.figure()
    final_img = store[-1][0].cpu().numpy()
    final_img_plot = plt.imshow(final_img, cmap='viridis')
    fig.colorbar(final_img_plot)
    plt.savefig(os.path.join(out_path, "final_sample.png"))
    plt.close(fig)

    # Plot the images with their timestep labels and add a color bar
    f, axarr = plt.subplots(1, len(store), figsize=(30, 5), gridspec_kw={'wspace': 0.05})
    
    if len(store) == 1:
        axarr = [axarr]
    
    for i in range(len(store)):
        # img = axarr[i].imshow(store[i].permute(1, 2, 0), vmin=vmin, vmax=vmax, cmap="viridis")  # Use a color map
        img = axarr[i].imshow(store[i].permute(1, 2, 0), cmap="viridis")  # Use a color map
        axarr[i].axis("off")
        axarr[i].set_title(f"Timestep {timesteps[i]}", fontsize=8)  # Add timestep as title

    # Add a color bar to the figure with adjusted size to match images
    cbar_ax = f.add_axes([0.92, 0.25, 0.02, 0.5])  # Adjusted to match image height and position
    f.colorbar(img, cax=cbar_ax)
    plt.savefig(os.path.join(out_path, "sample.png"), format='png')
    print(f'Saved at {out_path}!')

# Updated sampling function
def sample(model, noise_scheduler, out_path, device='cpu'):
    model.eval()
    sample = torch.randn(1, 1, 64, 64).to(device)

    store = []

    for i, t in enumerate(noise_scheduler.timesteps):
        if i % 1000 == 0: print(i, t)
        with torch.no_grad():
            # Convert t to a tensor and add a batch dimension
            t_tensor = torch.tensor([t], dtype=torch.int64).to(device)

            # Pass the timestep tensor with a batch dimension to the model
            residual = model(sample, t_tensor)

        # Update the sample using the noise scheduler
        # Convert the timestep t to an integer before passing to noise_scheduler.step()
        sample = noise_scheduler.step(residual, int(t), sample).prev_sample

        if i == 0 or (i + 1) % 100 == 0:
            store.append(sample.cpu().clone())  # Save for visualization

    # Plot the samples
    f, axarr = plt.subplots(1, len(store), figsize=(len(store) * 4, 4))
    for i in range(len(store)):
        axarr[i].imshow(store[i][0].permute(1, 2, 0), cmap='viridis')  # Assuming grayscale images
        axarr[i].axis("off")
    plt.savefig(out_path, format='png')
    print(f'Saved at {out_path}!')
    
    
# New sample function for end-to-end usage
def generate_sample(model_path, model, noise_scheduler, out_path, img_size=64, device='cpu'):
    # Load model
    model.load_state_dict(torch.load(model_path, map_location=device))

    # Generate sample
    sample_v1(model, noise_scheduler, out_path, device=device)

def get_constants(dataset_name='IllustrisTNG'):
    path = f'/groups/mlprojects/dm_diffusion/data/{dataset_name}_constants.pt'
    return torch.load(path)

# Global Normalization DM: 10.971004486083984, 0.5090954303741455
# Global Normalization Stellar: 0.11826974898576736, 1.0741989612579346
def denormalize(tensor, mean=10.971004486083984, std=0.5090954303741455): #TODO: make a lookup table, these are params for illustris
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

def save_images_unconditional(model, noise_scheduler, dir, epoch, device='cpu'):
    #samples 100 images to be plotted 
    sample = torch.randn(100, 1, 64, 64).to(device)

    for i, t in tqdm(enumerate(noise_scheduler.timesteps)):

        with torch.no_grad():
            residual = model(sample, t).sample

        sample = noise_scheduler.step(residual, t, sample).prev_sample

    path = os.path.join(dir, f"sample_imgs2_epoch{epoch}.pt")
    torch.save(sample, path)


def save_images_conditional(model, noise_scheduler, out_path, condition_loader=None, device='cpu'):
    model.eval()

    # Generate 100 samples
    for sample_idx in range(100):
        sample = torch.randn(1, 1, model.unet.sample_size, model.unet.sample_size).to(device)

        # Optionally set a conditional input
        guide = None
        if condition_loader:
            random_index = random.randint(0, len(condition_loader.dataset) - 1)
            guide = condition_loader.dataset[random_index].unsqueeze(0).to(device)
            if sample_idx == 0:  # Save guide only once
                torch.save(guide, os.path.join(out_path, 'guide.pt'))

        # Diffusion process
        for i, t in enumerate(noise_scheduler.timesteps):
            with torch.no_grad():
                residual = model(sample, t, guide)
            sample = noise_scheduler.step(residual, t, sample).prev_sample

            # Save only the last timestep sample as a .pt file
            if i == len(noise_scheduler.timesteps) - 1:
                torch.save(sample.cpu(), os.path.join(out_path, f'final_sample_{sample_idx}.pt'))


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
    image = image.to(torch.float32)
    noise = torch.normal(mean=0.0, std=sigma, size=image.shape, device=image.device)
    noisy_image = image + noise
    return noisy_image