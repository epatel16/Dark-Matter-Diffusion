import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt


# def sample(model, noise_scheduler, out_path, device='cpu'):
#   model.eval()
#   sample = torch.randn(1, 1, 64, 64).to(device)

#   store = []

#   for i, t in enumerate(noise_scheduler.timesteps):
#       print(i, t)
#       with torch.no_grad():
#           residual = model(sample, t).sample

#       sample = noise_scheduler.step(residual, t, sample).prev_sample

#       if i == 0 or (i + 1) % 100 == 0:
#         store.append(sample[0])

#   f, axarr = plt.subplots(1,11, figsize=(64, 64))
#   for i in range(len(store)):
#     axarr[i].imshow(denormalize(store[i]).cpu().permute(1, 2, 0))
#     axarr[i].axis("off")
#   plt.savefig(out_path, format='png')
#   print(f'Saved at {out_path}!')

# def sample(model, noise_scheduler, out_path, device='cpu'):
#     model.eval()
#     sample = torch.randn(1, 1, 64, 64).to(device)

#     store = []

#     for i, t in enumerate(noise_scheduler.timesteps):
#         print(i, t)
#         with torch.no_grad():
#             # Convert t to a tensor and add a batch dimension
#             t_tensor = torch.tensor([t], dtype=torch.int64).to(device)

#             # Pass the timestep tensor with a batch dimension to the model
#             residual = model(sample, t_tensor)

#         # Update the sample using the noise scheduler
#         # Convert the timestep t to an integer before passing to noise_scheduler.step()
#         sample = noise_scheduler.step(residual, int(t), sample).prev_sample

#         if i == 0 or (i + 1) % 100 == 0:
#             store.append(sample.cpu().clone())  # Save for visualization

#     # Plot the samples
#     f, axarr = plt.subplots(1, len(store), figsize=(64, 64))
#     for i in range(len(store)):
#         axarr[i].imshow(denormalize(store[i][0]).permute(1, 2, 0))  # Assuming denormalize reshapes properly
#         axarr[i].axis("off")
#     plt.savefig(out_path, format='png')
#     print(f'Saved at {out_path}!')

# Updated sampling function
def sample(model, noise_scheduler, out_path, device='cpu'):
    model.eval()
    sample = torch.randn(1, 1, 64, 64).to(device)

    store = []

    for i, t in enumerate(noise_scheduler.timesteps):
        if i % 100 == 0: print(i, t)
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
    sample(model, noise_scheduler, out_path, device=device)



class LogTransform:
  def __call__(self, x):
      return torch.log(x)  # Adding a small value to avoid log(0)

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