import os
import numpy as np
from torch import nn
from torch.utils.data import Dataset

from models.base import TGAFeatureExtractor

class AutoencoderDataset(Dataset):
    def __init__(self, DATA_DIR, transform=None, target_transform=None):
        self.X = None
        self.transform = transform
        self.target_transform = target_transform

        data = np.load(os.path.join(DATA_DIR, 'data.npz'))['TGA']
        self.X = data[:, 1:3, :] # Selecting only W and dWdT

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = np.array(self.X[idx])
        y = x.copy()
        if self.transform:
            x = self.transform(x)
        if self.target_transform:
            y = self.target_transform(y)
        return x, y
    


class ConvAutoencoder(TGAFeatureExtractor):
    def __init__(self):
        super().__init__()

        # encoder
        self.conv1 = nn.Sequential(nn.Conv1d(2, 32, 7, padding='same'), nn.LeakyReLU())
        self.pool1 = nn.MaxPool1d(2, 2, return_indices=True)
        self.conv2 = nn.Sequential(nn.Conv1d(32, 64, 7, padding='same'), nn.LeakyReLU())
        self.pool2 = nn.MaxPool1d(2, 2, return_indices=True)
        self.conv3 = nn.Sequential(nn.Conv1d(64, 28, 7, padding='same'), nn.LeakyReLU())
        self.pool3 = nn.MaxPool1d(2, 2, return_indices=True)
        self.flatten = nn.Flatten(1)
        self.fc1 = nn.Sequential(nn.Linear(3584, 64), nn.LeakyReLU())

        #decoder
        self.fc2 = nn.Sequential(nn.Linear(64, 3584), nn.LeakyReLU())
        self.unflatten = nn.Unflatten(1, (28, 128))
        self.unpool3 = nn.MaxUnpool1d(2, 2)
        self.conv4 = nn.Sequential(nn.Conv1d(28, 64, 7, padding='same'),nn.LeakyReLU())
        self.unpool2 = nn.MaxUnpool1d(2, 2)
        self.conv5 = nn.Sequential(nn.Conv1d(64, 32, 7, padding='same'), nn.LeakyReLU())
        self.unpool1 = nn.MaxUnpool1d(2, 2)
        self.conv6 = nn.Conv1d(32, 2, 7, padding='same')

    def encode(self, x):
        x = self.conv1(x)
        x, _ = self.pool1(x)
        x = self.conv2(x)
        x, _ = self.pool2(x)
        x = self.conv3(x)
        x, _ = self.pool3(x)
        x = self.flatten(x)
        x = self.fc1(x)

        return x

    def forward(self, x):
        x = self.conv1(x)
        x, indx1 = self.pool1(x)
        x = self.conv2(x)
        x, indx2 = self.pool2(x)
        x = self.conv3(x)
        x, indx3 = self.pool3(x)
        x = self.flatten(x)
        x = self.fc1(x)

        x = self.fc2(x)
        x = self.unflatten(x)
        x = self.unpool3(x, indx3)
        x = self.conv4(x)
        x = self.unpool2(x, indx2)
        x = self.conv5(x)
        x = self.unpool1(x, indx1)
        x = self.conv6(x)

        return x