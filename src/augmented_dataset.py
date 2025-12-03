import numpy as np
from scipy.signal import savgol_filter
from torch.utils.data import Dataset
import torch

class AugmentedTGADataset(Dataset):
    """
    Advanced TGA Augmentation.
    Includes:
    1. Temporal Shifting (Simulates kinetic variation)
    2. Vertical Scaling/Shifting (Simulates baseline drift/calibration)
    3. Gaussian Noise + Smoothing (Simulates sensor noise)
    """
    
    def __init__(self, data_path='./data/tga/data.npz', 
                 noise_std=0.015,           # 1.5% noise (Standard for Denoising VAEs)
                 shift_limit=30,            # ~30 degree shift (Aggressive but good for small data)
                 scale_limit=0.02,          # +/- 2% vertical scaling
                 savgol_window=15, 
                 savgol_poly=3, 
                 augmentation_factor=10,    
                 transform=None,
                 sample_indices=None,
                 mode='reconstruction'):      # NEW: Specify which samples to augment
        """
        Args:
            sample_indices: List/array of indices from the original dataset to augment.
                           If None, augments all samples in the dataset.
                           Use this to augment only training samples after train/test split.
            mode: 'reconstruction' (returns x, x) or 'feature' (returns x, y)
        """
        
        self.noise_std = noise_std
        self.shift_limit = shift_limit
        self.scale_limit = scale_limit
        self.savgol_window = savgol_window
        self.savgol_poly = savgol_poly
        self.augmentation_factor = augmentation_factor
        self.transform = transform
        self.mode = mode
        self.Y = None
        
        try:
            data = np.load(data_path)
            if 'TGA' in data:
                full_T = data['TGA'][:, 0, :]
                full_W = data['TGA'][:, 1, :]
                full_dWdT = data['TGA'][:, 2, :]
            elif 'X' in data:
                full_T = data['X'][:, 0, :]
                full_W = data['X'][:, 1, :]
                full_dWdT = data['X'][:, 2, :]
                if 'Y' in data:
                    self.Y = data['Y']
            else:
                raise ValueError("Unknown dataset format")
            
            if sample_indices is not None:
                self.T = full_T[sample_indices]
                self.W = full_W[sample_indices]
                self.dWdT = full_dWdT[sample_indices]
                if self.Y is not None:
                    self.Y = self.Y[sample_indices]
            else:
                self.T = full_T
                self.W = full_W
                self.dWdT = full_dWdT
                
        except FileNotFoundError:
            print("Warning: Data file not found, creating dummy sine waves.")
            B, L = 130, 1024
            self.T = np.linspace(30, 800, L)[None, :].repeat(B, 0)
            self.W = np.linspace(1, 0, L)[None, :].repeat(B, 0) 
            self.dWdT = np.zeros_like(self.W)
            
            if sample_indices is not None:
                self.T = self.T[sample_indices]
                self.W = self.W[sample_indices]
                self.dWdT = self.dWdT[sample_indices]

    def __len__(self):
        return self.W.shape[0] * self.augmentation_factor
    
    def apply_time_shift(self, signal, shift):
        """
        Shifts signal left or right. 
        """
        if shift == 0:
            return signal
        
        result = np.empty_like(signal)
        if shift > 0: # Shift Right (Delayed reaction)
            result[:shift] = signal[0]
            result[shift:] = signal[:-shift]
        else: # Shift Left (Earlier reaction)
            result[:shift] = signal[-shift:] 
            result[shift:] = signal[-1]      
            result[:(signal.shape[0] + shift)] = signal[-shift:]
            
        return result

    def get_scaling_params(self):
        """
        Generates consistent scaling parameters for a pair of W and dWdT.
        """
        # Multiplicative scale (Applies to both W and dWdT)
        scale = 1.0 + np.random.uniform(-self.scale_limit, self.scale_limit)
        
        # Additive offset (Applies ONLY to W, baseline shift doesn't affect slope)
        offset = np.random.uniform(-self.scale_limit, self.scale_limit)
        
        return scale, offset

    def add_gaussian_noise(self, signal):
        noise = np.random.normal(0, self.noise_std, signal.shape)
        # Note: We don't clip here yet, we smooth first, then clip in __getitem__ if needed
        return signal + noise
    
    def smooth_savgol(self, signal):
        return savgol_filter(signal, window_length=self.savgol_window, 
                           polyorder=self.savgol_poly, mode='nearest')
    
    def __getitem__(self, idx):
        original_idx = idx % self.W.shape[0]
        
        W_curr = self.W[original_idx].copy()
        dWdT_curr = self.dWdT[original_idx].copy()
        
        # 1. Temporal Shift (Apply SAME shift to both)
        shift = np.random.randint(-self.shift_limit, self.shift_limit)
        W_curr = self.apply_time_shift(W_curr, shift)
        dWdT_curr = self.apply_time_shift(dWdT_curr, shift)
        
        # 2. Scaling (Physically consistent)
        scale, offset = self.get_scaling_params()
        
        # Weight gets scale AND offset
        W_curr = W_curr * scale + offset
        
        # Derivative gets ONLY scale (derivative of the offset constant is 0)
        dWdT_curr = dWdT_curr * scale
        
        # Clip W to keep it roughly in [0, 1] (allowing slight overflow for noise is fine before clamping)
        W_curr = np.clip(W_curr, 0.0, 1.0)
        
        # 3. Noise + Smoothing
        W_aug = self.smooth_savgol(self.add_gaussian_noise(W_curr))
        dWdT_aug = self.smooth_savgol(self.add_gaussian_noise(dWdT_curr))
        
        # Final clip ensures inputs to Sigmoid-based VAE are strictly valid
        W_aug = np.clip(W_aug, 0.0, 1.0)
        
        # Stack
        x = np.stack([W_aug, dWdT_aug], axis=0)
        
        x_tensor = torch.from_numpy(x)
        
        if self.mode == 'reconstruction':
            return x_tensor, x_tensor.clone()
        elif self.mode == 'feature':
            if self.Y is None:
                raise ValueError("Labels (Y) not found in dataset, but mode='feature' requested.")
            return x_tensor, self.Y[original_idx]
        
        return x_tensor


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    dataset = AugmentedTGADataset(
        data_path='./data/tga/data.npz',
        noise_std=0.02,
        savgol_window=15,
        savgol_poly=3,
        augmentation_factor=5  # Generate 5x more data
    )
    
    print(f"Dataset size: {len(dataset)} (augmentation_factor={dataset.augmentation_factor})")
    
    # Visualize augmented samples
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    
    for i in range(3):
        x, _ = dataset[i]
        
        W = x[0].numpy() if torch.is_tensor(x) else x[0]
        dWdT = x[1].numpy() if torch.is_tensor(x) else x[1]
        
        # Plot W
        axes[i, 0].plot(W, linewidth=1.5)
        axes[i, 0].set_title(f'Sample {i} - Weight (W)')
        axes[i, 0].set_xlabel('Data Point')
        axes[i, 0].set_ylabel('W (normalized)')
        axes[i, 0].grid(True, alpha=0.3)
        
        # Plot dW/dT
        axes[i, 1].plot(dWdT, linewidth=1.5, color='orange')
        axes[i, 1].set_title(f'Sample {i} - Derivative (dW/dT)')
        axes[i, 1].set_xlabel('Data Point')
        axes[i, 1].set_ylabel('dW/dT (normalized)')
        axes[i, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figures/augmented_samples.png', dpi=150, bbox_inches='tight')
    print("Saved visualization to figures/augmented_samples.png")