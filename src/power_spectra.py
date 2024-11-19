import numpy as np
import matplotlib.pyplot as plt
import math

from torchvision import transforms
import torch
from torch.utils.data import DataLoader, Dataset

class NPYDataset(Dataset):
    def __init__(self, npy_file, num_samples, transform=None):
        self.data = np.load(npy_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample
    
def power(x):
    signal_ndim = x.dim() - 2
    signal_size = x.shape[-signal_ndim:]

    kmax = min(s for s in signal_size) // 2
    even = x.shape[-1] % 2 == 0

    x = torch.fft.rfftn(x, s=signal_size)  # new version broke BC
    x2 = x
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

def compute_pk(field, field_b=None,):
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

def dataloaderP(dataloader):
  P_samples = []
  counter = 0
  max = 100
  for img in dataloader:
    k, P = compute_pk((img[0]).unsqueeze(1).cpu().numpy())
    P_samples.append(P)
    counter += 1
    if counter == max:
      break
  P_stack = torch.stack(P_samples)
  mean_values = P_stack.mean(axis=0).cpu()
  return mean_values, P_samples

norms = [0.11826974898576736,
         1.0741989612579346,
         10.971004486083984,
         0.5090954303741455]

def samplesP(samples):
  P_samples = []
  for img in samples:
    k, P = compute_pk((10**(img * norms[1] + norms[0])).unsqueeze(1).cpu().numpy())
    P_samples.append(P)
  P_stack = torch.stack(P_samples)
  mean_values = P_stack.mean(axis=0).cpu()
  return mean_values, P_samples

npy_file = "/groups/mlprojects/dm_diffusion/data/Maps_Mcdm_IllustrisTNG_LH_z=0.00.npy"
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.RandomCrop((64, 64))
                                ])

Illustris_dataset = NPYDataset(npy_file, num_samples=100, transform=transform)
Illustris_dataloader = DataLoader(Illustris_dataset, batch_size=16, shuffle=True)

path = "/groups/mlprojects/dm_diffusion/power_spectra/sample_imgs2_epoch20.pt"
samples = torch.load(path)

boxsize = 6.25
k_conversion = 2*np.pi/boxsize

maps_mean, P_maps = dataloaderP(Illustris_dataloader)
sample_mean, P_sample = samplesP(samples)

k = torch.tensor(np.linspace(1, 32, num=32))
plt.figure(figsize=(12, 6))
for p in P_sample:
  plt.loglog(k.cpu(), p.cpu(), alpha = 0.025, color = 'blue')
plt.loglog(k.cpu(), sample_mean, label="Mean (Samples)", color = 'blue')
for p in P_maps:
  plt.loglog(k.cpu(), p.cpu(), alpha = 0.025, color = 'orange')
plt.loglog(k.cpu(), maps_mean, label = "Mean (Maps)", color = 'orange')
plt.xlabel("k")
plt.ylabel("P")
plt.legend()
plt.savefig("/groups/mlprojects/dm_diffusion/power_spectra/unconditional64x64.png")