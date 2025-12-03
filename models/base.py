import torch
import torch.nn as nn
from abc import ABC, abstractmethod

class TGAFeatureExtractor(nn.Module, ABC):
    """
    Abstract base class for all TGA feature extraction models.
    Enforces a common interface for encoding and saving/loading.
    """
    
    @abstractmethod
    def forward(self, x):
        """
        Forward pass for training (usually reconstruction).
        """
        pass

    @abstractmethod
    def encode(self, x):
        """
        Returns the latent vector (feature encoding).
        """
        pass

    def compute_loss(self, output, target, **kwargs):
        """
        Computes the loss for the model.
        Handles standard reconstruction (MSE) and VAE (MSE + KLD) if output is a tuple.
        """
        if isinstance(output, tuple) and len(output) == 3:
            recon_x, mu, logvar = output
            criterion = nn.MSELoss()
            mse_loss = criterion(recon_x, target)
            
            # KL Divergence
            kld_weight = kwargs.get('kld_weight', 0.0)
            kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            kld_loss /= recon_x.size(0)
            
            return mse_loss + kld_weight * kld_loss
            
        criterion = nn.MSELoss()
        return criterion(output, target)
    
    def save_checkpoint(self, path, optimizer=None, epoch=None, loss=None):
        """Standardized saving."""
        state = {
            'model_state_dict': self.state_dict(),
            'config': self.config if hasattr(self, 'config') else {}
        }
        if optimizer:
            state['optimizer_state_dict'] = optimizer.state_dict()
        if epoch is not None:
            state['epoch'] = epoch
        if loss is not None:
            state['loss'] = loss
            
        torch.save(state, path)

    def load_checkpoint(self, path, device):
        """Standardized loading."""
        # Fix: Explicitly set weights_only=False to support legacy checkpoints
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        
        # Handle both full checkpoint dicts and simple state_dicts
        if 'model_state_dict' in checkpoint:
            self.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.load_state_dict(checkpoint)
        return checkpoint