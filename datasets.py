import torch
from torch._C import dtype
from torch.utils.data import Dataset
import numpy as np
import os
import glob
from utils import DownSampler


class SRDataset(Dataset):
    """
    A PyTorch Dataset to be used by a PyTorch DataLoader.
    """

    def __init__(self, data_path, scaling_factor):
        self.scaling_factor = scaling_factor
        self.data_files = sorted(glob.glob(os.path.join(data_path, "*.npy")))
        self.down_scaler = DownSampler(scaling_factor)

    def __getitem__(self, i):
        """
        This method is required to be defined for use in the PyTorch DataLoader.
        :param i: index to retrieve
        :return: the 'i'th pair LR and HR images to be fed into the model
        """
        # Read npy
        arr = np.load(self.data_files[i])
        upper_layer = arr[0, :, :]
        std = upper_layer.flatten().std()
        avg = upper_layer.flatten().mean()
        normalized = (upper_layer - avg) / std
        high_res = torch.tensor(normalized, dtype=torch.float32) # upper layer
        low_res = self.down_scaler(high_res)

        return low_res, high_res

    def __len__(self):
        """
        This method is required to be defined for use in the PyTorch DataLoader.
        :return: size of this data (in number of images)
        """
        return len(self.data_files)