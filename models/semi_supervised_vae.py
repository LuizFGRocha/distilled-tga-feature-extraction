import torch
import torch.nn as nn
import torch.nn.functional as F
from models.even_smaller_variational_autoencoder import EvenSmallerVAE

class SemiSupervisedVAE(EvenSmallerVAE):
    """
    Semi-supervised VAE that can be trained with both labeled and unlabeled data.
    
    Training Strategy:
    1. Pretrain on all unlabeled data (unsupervised VAE)
    2. Fine-tune with supervised loss on labeled subset
    """
    
    def __init__(self, compressed_dim=8, dropout_prob=0.2, num_targets=25):
        super().__init__(compressed_dim, dropout_prob)
        
        # Prediction head for supervised learning
        # Maps from latent space to AFM targets
        self.prediction_head = nn.Sequential(
            nn.Linear(compressed_dim, 64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.Linear(32, num_targets)
        )
        
        # Weight for supervised loss (can be adjusted)
        self.supervised_weight = 0.0  # Start with 0 for unsupervised pretraining
        
    def forward(self, x):
        """Standard VAE forward pass"""
        reconstruction, mu, logvar = super().forward(x)
        return reconstruction, mu, logvar
    
    def predict(self, x):
        """Predict AFM targets from input"""
        latent = self.encode(x)
        return self.prediction_head(latent)
    
    def compute_supervised_loss(self, x, y_targets):
        """
        Compute supervised prediction loss.
        
        Args:
            x: Input TGA curves
            y_targets: Ground truth AFM measurements (B, 10)
                      Flattened from (B, 5, 2) -> (B, 10)
        """
        predictions = self.predict(x)
        return F.mse_loss(predictions, y_targets)
    
    def set_supervised_weight(self, weight):
        """Set the weight for supervised loss component"""
        self.supervised_weight = weight
        
    def enable_supervised_learning(self, weight=0.5):
        """Enable supervised learning mode with specified weight"""
        self.supervised_weight = weight
        print(f"Supervised learning enabled with weight: {weight}")
        
    def disable_supervised_learning(self):
        """Disable supervised learning (pure VAE mode)"""
        self.supervised_weight = 0.0
        print("Supervised learning disabled (pure VAE mode)")
