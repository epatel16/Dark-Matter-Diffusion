import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
import torchvision
from torchvision import transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import os
import io
import boto3


class S3PairedDataset(Dataset):
    def __init__(self, 
                 s3: object,
                 bucket_name: str,
                 keys: list[str],
                 transforms: list[callable],
                 cache_dir: str = "/tmp/s3_cache",
                 aws_region: str | None = None):
        """
        bucket_name: your S3 bucket name
        keys:       list of S3 object keys, e.g. ['Maps_Mcdm_Astrid_LH_z=0.00.npy', ...]
        transforms: list of torchvision‐style transforms, one per key
        """
        assert len(keys) == len(transforms), "keys and transforms must align"
        os.makedirs(cache_dir, exist_ok=True)

        self.bucket     = bucket_name
        self.keys       = keys
        self.transforms = transforms

        self.s3 = s3

        self.data = []
        for key in keys:
            local_path = os.path.join(cache_dir, os.path.basename(key))
            if not os.path.exists(local_path):
                # download once
                s3.download_file(bucket_name, key, local_path)
            # open as memmap — no full read into RAM
            arr = np.load(local_path, mmap_mode='r')  # shape (N, H, W)
            self.data.append(arr)

        self.transforms = transforms


    def __len__(self):
        # assume all arrays have same leading dimension
        return self.data[0].shape[0]

    def __getitem__(self, idx: int):
        # grab the idx-th slice from each array
        samples = [d[idx] for d in self.data]          
        # to torch.Tensor with channel dim
        samples = [torch.tensor(s, dtype=torch.float32).unsqueeze(0)
                   for s in samples]
        # apply per-channel transform
        if self.transforms:
            samples = [t(s) for s, t in zip(samples, self.transforms)]
        return samples


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


class GravitationalLensDataset(Dataset):
    def __init__(self, dm_file, shear_map_dir, dataset, transform_dm=None, transform_cond=None):

        self.dm_data = np.load(dm_file, mmap_mode='r')
        self.shear_map_dir = shear_map_dir
        self.transform_dm = transform_dm
        self.transform_cond = transform_cond
        self.dataset = dataset

        self.chunk_size = 3000
        self.num_batches = len([f for f in os.listdir(shear_map_dir) if f.startswith("g1")])
        self.dataset_length = min(len(self.dm_data), self.num_batches * self.chunk_size)

        self.g1_batch = None
        self.g2_batch = None

    def __len__(self):
        return self.dataset_length

    def __getitem__(self, idx):
        batch_idx = idx // self.chunk_size * self.chunk_size
        offset = idx % self.chunk_size

        if idx == batch_idx:
            g1_path = os.path.join(self.shear_map_dir, f"g1_{self.dataset}_{batch_idx}.npy")
            g2_path = os.path.join(self.shear_map_dir, f"g2_{self.dataset}_{batch_idx}.npy")
            self.g1_batch = np.load(g1_path)
            self.g2_batch = np.load(g2_path)

        conditioning_map = np.stack([self.g1_batch[offset], self.g2_batch[offset]], axis=0)
        dm_map = self.dm_data[idx]

        dm_map = torch.tensor(dm_map, dtype=torch.float32).unsqueeze(0)
        conditioning_map = torch.tensor(conditioning_map, dtype=torch.float32)

        if self.transform_dm:
            dm_map = self.transform_dm(dm_map)
        if self.transform_cond:
            conditioning_map = self.transform_cond(conditioning_map)

        return dm_map, conditioning_map


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
