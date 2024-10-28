import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt


def sample(model, noise_scheduler, out_path, device='cpu'):
  model.eval()
  sample = torch.randn(1, 1, 64, 64).to(device)

  store = []

  for i, t in enumerate(noise_scheduler.timesteps):
      print(i, t)
      with torch.no_grad():
          residual = model(sample, t).sample

      sample = noise_scheduler.step(residual, t, sample).prev_sample

      if i == 0 or (i + 1) % 100 == 0:
        store.append(sample[0])

  f, axarr = plt.subplots(1,11, figsize=(64, 64))
  for i in range(len(store)):
    axarr[i].imshow(denormalize(store[i]).cpu().permute(1, 2, 0))
    axarr[i].axis("off")
  plt.savefig(out_path, format='png')
  print(f'Saved at {out_path}!')

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