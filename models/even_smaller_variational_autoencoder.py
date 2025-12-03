import torch
import torch.nn as nn
from models.base import TGAFeatureExtractor

class EvenSmallerVAE(TGAFeatureExtractor):
    def __init__(self, compressed_dim=8, dropout_prob=0.2):
        super().__init__()
        
        self.encoder_backbone = nn.Sequential(
            nn.Conv1d(2, 16, kernel_size=7, stride=2, padding=3), # -> (16, 512)
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_prob),
            
            # Layer 2
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2), # -> (32, 256)
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_prob),
            
            # Layer 3
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2), # -> (64, 128)
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_prob),
        )

        self.channel_bottleneck = nn.Conv1d(64, 4, kernel_size=3, stride=1, padding=1)
        
        self.flat_dim = 4 * 128 
        
        self.fc_mu = nn.Linear(self.flat_dim, compressed_dim)
        self.fc_var = nn.Linear(self.flat_dim, compressed_dim)
        
        self.decoder_input = nn.Linear(compressed_dim, self.flat_dim)
        
        self.channel_expand = nn.Conv1d(4, 64, kernel_size=3, stride=1, padding=1)
        
        self.decoder_backbone = nn.Sequential(
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1), # -> (32, 256)
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_prob),
            
            nn.ConvTranspose1d(32, 16, kernel_size=4, stride=2, padding=1), # -> (16, 512)
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_prob),
            
            nn.ConvTranspose1d(16, 2, kernel_size=4, stride=2, padding=1), # -> (2, 1024)
            nn.Sigmoid() 
        )

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def forward(self, x):
        h = self.encoder_backbone(x)    # (B, 64, 128)
        
        h = self.channel_bottleneck(h)  # (B, 4, 128)
        
        flat = h.view(h.size(0), -1)    # (B, 512)
        
        mu = self.fc_mu(flat)
        logvar = self.fc_var(flat)
        z = self.reparameterize(mu, logvar)
        
        d = self.decoder_input(z)       # (B, 512)
        d = d.view(-1, 4, 128)          # (B, 4, 128)
        d = self.channel_expand(d)      # (B, 64, 128)
        
        reconstruction = self.decoder_backbone(d)
        
        return reconstruction, mu, logvar

    def encode(self, x):
        h = self.encoder_backbone(x)    # (B, 64, 128)
        h = self.channel_bottleneck(h)  # (B, 4, 128)
        flat = h.view(h.size(0), -1)    # (B, 512)
        return self.fc_mu(flat)
