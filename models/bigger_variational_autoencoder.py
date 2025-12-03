import torch
import torch.nn as nn
import os 
import sys

# add project root to path to import from scripts/
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.base import TGAFeatureExtractor

class Bigger_VAE(TGAFeatureExtractor):
    def __init__(self, compressed_dim=64):
        super().__init__()
        
        # Shared Encoder layers
        # Input: (Batch, 2, 1024)
        self.conv_layers = nn.Sequential(
            nn.Conv1d(2, 64, 4, stride=2, padding=1),    # -> (64, 512)
            nn.ReLU(),
            nn.Conv1d(64, 128, 4, stride=2, padding=1),  # -> (128, 256)
            nn.ReLU(),
            nn.Conv1d(128, 256, 4, stride=2, padding=1), # -> (256, 128)
            nn.ReLU(),
            nn.Conv1d(256, 512, 4, stride=2, padding=1), # -> (512, 64)
            nn.ReLU(),
            nn.Conv1d(512, 1024, 4, stride=2, padding=1),# -> (1024, 32)
            nn.ReLU(),
            nn.Flatten()
        )
        
        # VAE specific heads
        # Flatten size: 1024 channels * 32 length = 32768
        self.fc_mu = nn.Linear(1024 * 32, compressed_dim)
        self.fc_var = nn.Linear(1024 * 32, compressed_dim)
        
        # Decoder
        self.decoder_input = nn.Linear(compressed_dim, 1024 * 32)
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(1024, 512, 4, stride=2, padding=1), # -> (512, 64)
            nn.ReLU(),
            nn.ConvTranspose1d(512, 256, 4, stride=2, padding=1),  # -> (256, 128)
            nn.ReLU(),
            nn.ConvTranspose1d(256, 128, 4, stride=2, padding=1),  # -> (128, 256)
            nn.ReLU(),
            nn.ConvTranspose1d(128, 64, 4, stride=2, padding=1),   # -> (64, 512)
            nn.ReLU(),
            nn.ConvTranspose1d(64, 2, 4, stride=2, padding=1)      # -> (2, 1024)
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x):
        # For evaluation, we typically use the mean (mu) as the encoding
        x = self.conv_layers(x)
        return self.fc_mu(x)

    def forward(self, x):
        flat = self.conv_layers(x)
        mu = self.fc_mu(flat)
        logvar = self.fc_var(flat)
        
        z = self.reparameterize(mu, logvar)
        
        d = self.decoder_input(z)
        d = d.view(-1, 1024, 32)
        reconstruction = self.decoder(d)
        
        # Return tuple for custom loss handling
        return reconstruction, mu, logvar

if __name__ == "__main__":
    model = Bigger_VAE()
    print(f"Model parameter count: {sum(p.numel() for p in model.parameters())}")