import numpy as np
import torch
from matplotlib import pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from utils import get_constants
import argparse
import os

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
  return k, mean_values, P_samples

def samplesP(samples, std, mean):
  P_samples = []
  for img in samples:
    k, P = compute_pk(10**(img * std + mean).unsqueeze(1).cpu().numpy())
    P_samples.append(P)
  P_stack = torch.stack(P_samples)
  mean_values = P_stack.mean(axis=0).cpu()
  return k, mean_values, P_samples

def gtP(gt):
  k, P = compute_pk(gt.cpu().numpy())
  return k, P, P

def parse_args():
    parser = argparse.ArgumentParser(description="A script to parse training parameters.")
    
    # Adding arguments
    parser.add_argument("--dataset", default="IllustrisTNG", type=str, required=False, help="IllustrisTNG, Astrid, or SIMBA.")
    #parser.add_argument("--condition", default="star", type=str, required=False, help="star or noise")
    #parser.add_argument("--dir", default="100diff", type=str, required=False, help="100diff or 100same")  

    # Parsing arguments
    return parser.parse_args()

def power_spectra(args):

  path = "/groups/mlprojects/dm_diffusion/model_out/FinalExps/lr0.0001_step1000_size64_condTrue/stellar_0.1/20241205_235509/ep29/"
   
  constants = get_constants(dataset_name = args.dataset)

  #npy_file = f"/groups/mlprojects/dm_diffusion/data/Maps_Mcdm_{args.dataset}_LH_z=0.00.npy"
  #transform = transforms.Compose([transforms.ToTensor()])
  
  #dataset = NPYDataset(npy_file, num_samples=100, transform=transform)
  #dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
  
  #samples = torch.load(f"/groups/mlprojects/dm_diffusion/final_conditional/{args.dataset}/{args.condition}/{args.dir}/sample.pt", map_location=torch.device('cpu'))
  #guides = torch.load(f"/groups/mlprojects/dm_diffusion/final_conditional/{args.dataset}/{args.condition}/{args.dir}/guide.pt", map_location=torch.device('cpu'))
  # k_gt, gt_mean, P_gt = dataloaderP(dataloader)
  gt = torch.load(path + "gt.pt")
  #sample_files = [os.path.join(path, f"sample_{i}.pt") for i in range(10)]
  #tensors = [torch.load(file) for file in sample_files]
  #samples = torch.stack(tensors)
  samples = torch.load(path + "sample999.pt")

  k_gt, gt_mean, P_gt = gtP(gt)
  k_sample, sample_mean, P_sample = samplesP(samples, constants['dm_std'], constants['dm_mean'])

  plt.figure(figsize=(12, 6))
  for p in P_sample:
    plt.loglog(k_sample.cpu(), p.cpu(), alpha = 0.025, color = 'orange')
  plt.loglog(k_sample.cpu(), sample_mean, label="Mean (Conditional Samples)", color = 'orange')
  #for p in P_gt:
  #  plt.loglog(k_gt.cpu(), p.cpu(), alpha = 0.025, color = 'blue')
  plt.loglog(k_gt.cpu(), gt_mean, label = "Mean (GT Maps)", color = 'blue')
  #for p in P_guide:
  #  plt.loglog(k_guide.cpu(), p.cpu(), alpha = 0.025, color = 'green')
  #plt.loglog(k_guide.cpu(), guide_mean, label = "Mean (Guide Maps)", color = 'green')
  plt.xlabel("k")
  plt.ylabel("P")
  plt.title(args.dataset)
  plt.legend()
  plt.savefig(f"{path}/power_spectra.png")

def corr_R(args):

  path = "/groups/mlprojects/dm_diffusion/model_out/FinalExps/lr0.0001_step1000_size64_condTrue/stellar_0.1/20241205_235509/ep29/"
  constants = get_constants(dataset_name = args.dataset)

  #gts = torch.load(f"/groups/mlprojects/dm_diffusion/final_conditional/{args.dataset}/{args.condition}/{args.dir}/gt.pt", map_location=torch.device('cpu'))
  #samples = torch.load(f"/groups/mlprojects/dm_diffusion/final_conditional/{args.dataset}/{args.condition}/{args.dir}/sample.pt", map_location=torch.device('cpu'))
  
  gts = torch.load(path + "gt.pt")
  samples = torch.load(path + "sample999.pt")
  print(samples)

  #bs = 25
  bs = 6.25
  R_samples = []
  for i in range(100):
    #gt = (10**gts[i]).unsqueeze(1).cpu().numpy()
    gt = (10**gts).cpu().numpy()
    sample = (10**(samples[i] * constants['dm_std'] + constants['dm_mean'])).unsqueeze(1).cpu().numpy()
    k = compute_pk(gt, field_b=sample, boxsize=bs)[0]
    corr = compute_pk(gt, field_b=sample, boxsize=bs)[1]/(np.sqrt(compute_pk(gt, boxsize=bs)[1]) * np.sqrt(compute_pk(sample, boxsize=bs)[1]))
    R_samples.append(corr)
  R_samples = np.array(R_samples)

  plt.figure(figsize=(12, 6))
  plt.plot(k, np.mean(R_samples, axis=0), label = "mean(R(k))")
  plt.fill_between(k, np.percentile(R_samples, 10, axis=0), np.percentile(R_samples, 90, axis=0), color="blue", alpha=0.2, label="Spread (10th pct - 90 pct)")
  plt.xlabel("k")
  plt.ylabel("R(k)")
  plt.ylim(0, 1)
  plt.legend(loc = 'lower left')
  plt.title(args.dataset)
  plt.savefig(f"{path}/power_spectra/corr.png")

if __name__ == "__main__":
    args = parse_args()
    power_spectra(args)
    corr_R(args)