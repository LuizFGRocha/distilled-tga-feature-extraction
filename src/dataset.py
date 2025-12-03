import os
import numpy as np
import torch
from torch.utils.data import Dataset

class TGADataset(Dataset):
    def __init__(self, data_path='./data/tga/data.npz', transform=None, mode='reconstruction'):
        """
        mode: 'reconstruction' (returns x, x) or 'feature' (returns x, y_stats)
        """
        self.transform = transform
        self.mode = mode
        
        data = np.load(data_path)
        
        # Handle different data structures (tga vs tga_afm)
        if 'TGA' in data:
            self.X = data['TGA'][:, 1:3, :] # W and dW/dT
            self.Y = None
        elif 'X' in data:
            self.X = data['X'][:, 1:3, :]
            self.Y = data['Y']
        else:
            raise ValueError("Unknown dataset format")

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = np.array(self.X[idx])
        
        if self.transform:
            x = self.transform(x)
        else:
            x = torch.tensor(x)

        if self.mode == 'reconstruction':
            return x, x
        elif self.mode == 'feature':
            return x, self.Y[idx]
        return x