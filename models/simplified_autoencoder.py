import torch
from torch import nn

class SimpleAutoencoder(nn.Module):
    """
    A very simple fully-connected autoencoder.
    This is a much more appropriate architecture for a very small dataset (e.g., 120 samples)
    than a complex U-Net, as it has far fewer parameters.
    """
    def __init__(self, ch_in=2, length=1024, compressed_dim=3, intermediate_dim=16):
        super(SimpleAutoencoder, self).__init__()
        
        self.input_channels = ch_in
        self.input_length = length
        input_flat_dim = ch_in * length # 2 * 1024 = 2048
        
        # 2048 -> 16 -> 3
        self.encoder = nn.Sequential(
            nn.Linear(input_flat_dim, intermediate_dim),
            nn.ReLU(inplace=True),
            nn.Linear(intermediate_dim, compressed_dim) # The compressed encoding
        )
        
        # 3 -> 16 -> 2048
        self.decoder = nn.Sequential(
            nn.Linear(compressed_dim, intermediate_dim),
            nn.ReLU(inplace=True),
            nn.Linear(intermediate_dim, input_flat_dim)
            # No final activation (e.g., Sigmoid) as we want to reconstruct
            # the original signal values, not just 0-1.
        )
        
    def encode(self, x):
        # 1. Flatten the input
        # Input shape: (batch_size, channels, length) -> (batch_size, channels * length)
        x_flat = x.view(x.size(0), -1)
        
        # 2. Pass through encoder
        # (batch_size, channels * length) -> (batch_size, compressed_dim)
        compressed = self.encoder(x_flat)
        return compressed

    def forward(self, x):
        # 1. Encode
        # (batch_size, channels, length) -> (batch_size, compressed_dim)
        compressed = self.encode(x)
        
        # 2. Decode
        # (batch_size, compressed_dim) -> (batch_size, channels * length)
        decompressed_flat = self.decoder(compressed)
        
        # 3. Reshape back to original image shape
        # (batch_size, channels * length) -> (batch_size, channels, length)
        reconstruction = decompressed_flat.view(x.size(0), self.input_channels, self.input_length)
        
        return reconstruction

if __name__ == '__main__':
    # Example usage:
    # Input shape: (batch_size, channels, length)
    # Batch of 4 samples, 2 channels, 1024 length
    test_input = torch.randn(4, 2, 1024)
    
    # Initialize the simple model
    # It will produce a 3-dimensional encoding
    model = SimpleAutoencoder(ch_in=2, length=1024, compressed_dim=3)
    
    # Test the forward pass (reconstruction)
    reconstruction = model(test_input)
    
    # Test the encode pass
    encoding = model.encode(test_input)
    
    print(f"--- SimpleAutoencoder Test ---")
    print(f"Original input shape: {test_input.shape}")
    print(f"Reconstruction shape: {reconstruction.shape}")
    print(f"Encoding shape:       {encoding.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")
    print("\nThis parameter count is MUCH lower and more suitable for 120 data points.")