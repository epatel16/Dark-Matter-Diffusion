import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
import torchvision
from torchvision import transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class NPYDataset(Dataset): 
    def __init__(self, npy_file, transform=None):
        self.data = np.load(npy_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample
        
class PairedDataset(Dataset):
    def __init__(self, file_paths, transform=None):
        self.file_paths = file_paths
        self.data = [np.load(file_path, mmap_mode='r') for file_path in file_paths]  # Memory map to save memory
        self.transform = transform

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, idx):
        sample = [d[idx] for d in self.data]
        sample = [torch.tensor(s, dtype=torch.float32).unsqueeze(0) for s in sample]
        if self.transform:
            sample = [t(s) for s, t in zip(sample, self.transform)]
        return sample

class SlicedDataset(Dataset):
    def __init__(self, file_path, transform=None):
        self.file_path = file_path
        self.data = np.load(file_path, mmap_mode='r')  # Memory map to save memory
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        sample = torch.tensor(sample, dtype=torch.float32).unsqueeze(0)
        if self.transform:
            sample = self.transform(sample)
        return sample