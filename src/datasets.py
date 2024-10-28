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

# class SlicedDataset(Dataset):
#     def __init__(self, npy_file, num_samples, transform=None):
#         # Using numpy.memmap to load only parts of the data as needed
#         self.data = np.memmap(npy_file, dtype='float32', mode='r')  # Adjust dtype if needed
#         self.num_samples = num_samples
#         self.transform = transform

#     def __len__(self):
#         return self.num_samples

#     def __getitem__(self, idx):
#         start_idx = idx * self.num_samples
#         end_idx = start_idx + self.num_samples
#         sample = self.data[start_idx:end_idx]

#         if self.transform:
#             sample = self.transform(sample)

#         return sample

class SlicedDataset(Dataset):
    def __init__(self, file_path, transform=None):
        self.file_path = file_path
        self.data = np.load(file_path, mmap_mode='r')  # Memory-mapped to avoid loading everything at once
        self.transform = transform  # Apply transformations like normalization

    def __len__(self):
        return len(self.data) 

    def __getitem__(self, idx):
        sample = self.data[idx]
        if self.transform:
            sample = self.transform(sample)  # Apply transformation
        
        return sample