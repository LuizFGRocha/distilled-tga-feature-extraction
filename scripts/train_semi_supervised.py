"""
Semi-Supervised VAE Training Script

Two-stage training:
1. Unsupervised pretraining on all TGA data (113 + 33 = 146 samples)
2. Supervised fine-tuning on labeled AFM data (33 samples only)

Usage:
    python scripts/train_semi_supervised.py --latent_dim 16 --supervised_weight 0.3
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, ConcatDataset
from torch.utils.tensorboard import SummaryWriter
import os
import sys
import numpy as np
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import TGADataset
from src.augmented_dataset import AugmentedTGADataset
from models.semi_supervised_vae import SemiSupervisedVAE
from src.visualization import visualize_reconstructions


def load_datasets(args):
    """Load both unlabeled and labeled datasets"""
    
    # Unlabeled TGA data (113 samples)
    unlabeled_dataset = TGADataset(
        data_path='./data/tga/data.npz', 
        mode='reconstruction'
    )
    
    # Labeled TGA+AFM data (33 samples)
    labeled_dataset = TGADataset(
        data_path='./data/tga_afm/data.npz',
        mode='feature'
    )
    
    print(f"Unlabeled dataset: {len(unlabeled_dataset)} samples")
    print(f"Labeled dataset: {len(labeled_dataset)} samples")
    
    return unlabeled_dataset, labeled_dataset


def prepare_dataloaders(unlabeled_dataset, labeled_dataset, args):
    """
    Prepare train/test splits for both stages:
    - Stage 1 (unsupervised): uses all data
    - Stage 2 (supervised): uses only labeled data
    """
    
    # Split labeled data into train/test (80/20)
    labeled_train_size = int(0.8 * len(labeled_dataset))
    labeled_test_size = len(labeled_dataset) - labeled_train_size
    
    labeled_train, labeled_test = random_split(
        labeled_dataset, 
        [labeled_train_size, labeled_test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"Labeled train: {labeled_train_size}, Labeled test: {labeled_test_size}")
    
    # For stage 1: combine all unlabeled + labeled (as reconstruction task)
    # Convert labeled data to reconstruction mode
    labeled_for_unsupervised = TGADataset(
        data_path='./data/tga_afm/data.npz',
        mode='reconstruction'
    )
    
    # Split labeled_for_unsupervised matching labeled_train indices
    labeled_unsup_train, labeled_unsup_test = random_split(
        labeled_for_unsupervised,
        [labeled_train_size, labeled_test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Combine all data for unsupervised pretraining
    unsupervised_train = ConcatDataset([unlabeled_dataset, labeled_unsup_train])
    
    print(f"Stage 1 (unsupervised) train size: {len(unsupervised_train)}")
    print(f"Stage 2 (supervised) train size: {labeled_train_size}")
    
    # Data augmentation for stage 1
    if args.use_augmentation:
        print(f"Augmenting unsupervised training data (factor={args.augmentation_factor})...")
        
        # Get indices for augmentation
        unlabeled_indices = list(range(len(unlabeled_dataset)))
        labeled_indices = labeled_unsup_train.indices
        
        # Create augmented datasets
        aug_unlabeled = AugmentedTGADataset(
            data_path='./data/tga/data.npz',
            noise_std=args.noise_std,
            savgol_window=args.savgol_window,
            savgol_poly=args.savgol_poly,
            augmentation_factor=args.augmentation_factor,
            sample_indices=unlabeled_indices
        )
        
        aug_labeled = AugmentedTGADataset(
            data_path='./data/tga_afm/data.npz',
            noise_std=args.noise_std,
            savgol_window=args.savgol_window,
            savgol_poly=args.savgol_poly,
            augmentation_factor=args.augmentation_factor,
            sample_indices=labeled_indices
        )
        
        unsupervised_train = ConcatDataset([aug_unlabeled, aug_labeled])
        print(f"Augmented train size: {len(unsupervised_train)}")
    
    # Create dataloaders
    unsupervised_train_loader = DataLoader(
        unsupervised_train,
        batch_size=args.batch_size,
        shuffle=True
    )
    
    # Supervised training dataset (Augmented or Original)
    if args.use_augmentation:
        print(f"Augmenting supervised training data (factor={args.augmentation_factor})...")
        supervised_train_ds = AugmentedTGADataset(
            data_path='./data/tga_afm/data.npz',
            noise_std=args.noise_std,
            savgol_window=args.savgol_window,
            savgol_poly=args.savgol_poly,
            augmentation_factor=args.augmentation_factor,
            sample_indices=labeled_train.indices,
            mode='feature'
        )
    else:
        supervised_train_ds = labeled_train

    supervised_train_loader = DataLoader(
        supervised_train_ds,
        batch_size=args.batch_size,
        shuffle=True
    )
    
    supervised_test_loader = DataLoader(
        labeled_test,
        batch_size=labeled_test_size,
        shuffle=False
    )
    
    # Test loader for reconstruction visualization
    test_recon_loader = DataLoader(
        labeled_unsup_test,
        batch_size=args.batch_size,
        shuffle=False
    )
    
    return (unsupervised_train_loader, supervised_train_loader, 
            supervised_test_loader, test_recon_loader, labeled_train)


def compute_target_statistics(supervised_train_loader, device):
    """Compute mean and std of targets for normalization"""
    all_targets = []
    
    for _, y in supervised_train_loader:
        # y shape: (B, 5, 2) -> flatten to (B, 10)
        y_flat = y.view(y.size(0), -1)
        all_targets.append(y_flat)
    
    all_targets = torch.cat(all_targets, dim=0).double()
    
    y_mean = all_targets.mean(dim=0).to(device)
    y_std = all_targets.std(dim=0).to(device)
    
    # Avoid division by zero
    y_std[y_std < 1e-8] = 1.0
    
    print(f"Target statistics computed from {all_targets.size(0)} samples")
    
    return y_mean, y_std


def train_stage1_unsupervised(model, train_loader, test_loader, args, device, writer):
    """
    Stage 1: Unsupervised pretraining on all data
    Pure VAE training for reconstruction
    """
    print("\n" + "="*80)
    print("STAGE 1: UNSUPERVISED PRETRAINING")
    print("="*80)
    
    model.disable_supervised_learning()
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    
    global_step = 0
    best_test_loss = float('inf')
    
    for epoch in range(args.stage1_epochs):
        model.train()
        train_loss = 0
        
        for x, target in train_loader:
            x, target = x.to(device), target.to(device)
            
            optimizer.zero_grad()
            
            reconstruction, mu, logvar = model(x)
            loss = model.compute_loss(
                (reconstruction, mu, logvar), 
                target, 
                kld_weight=args.kld_weight
            )
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            global_step += 1
            
            if global_step % 100 == 0:
                print(f"[Stage 1] Epoch [{epoch+1}/{args.stage1_epochs}], "
                      f"Step [{global_step}], Loss: {loss.item():.6f}")
        
        avg_train_loss = train_loss / len(train_loader)
        writer.add_scalar('Stage1/train_loss', avg_train_loss, epoch)
        
        # Validation
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for x, target in test_loader:
                x, target = x.to(device), target.to(device)
                
                reconstruction, mu, logvar = model(x)
                loss = model.compute_loss(
                    (reconstruction, mu, logvar),
                    target,
                    kld_weight=args.kld_weight
                )
                test_loss += loss.item()
        
        avg_test_loss = test_loss / len(test_loader)
        writer.add_scalar('Stage1/test_loss', avg_test_loss, epoch)
        
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
        
        if (epoch + 1) % 100 == 0:
            print(f"[Stage 1] Epoch {epoch+1}: Train Loss = {avg_train_loss:.6f}, "
                  f"Test Loss = {avg_test_loss:.6f}")
    
    print(f"\n[Stage 1] Complete! Best test loss: {best_test_loss:.6f}\n")
    return model


def train_stage2_supervised(model, train_loader, test_loader, y_mean, y_std, 
                           args, device, writer, save_dir):
    """
    Stage 2: Supervised fine-tuning on labeled data
    Combined VAE + prediction loss
    """
    print("\n" + "="*80)
    print("STAGE 2: SUPERVISED FINE-TUNING")
    print("="*80)
    
    model.enable_supervised_learning(weight=args.supervised_weight)
    
    # Use smaller learning rate for fine-tuning
    optimizer = optim.Adam(model.parameters(), lr=args.lr * 0.1, weight_decay=1e-5)
    
    global_step = 0
    best_test_loss = float('inf')
    
    for epoch in range(args.stage2_epochs):
        model.train()
        train_recon_loss = 0
        train_pred_loss = 0
        train_total_loss = 0
        
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            
            # Flatten and normalize targets
            y_flat = y.view(y.size(0), -1).double()
            y_norm = (y_flat - y_mean) / y_std
            
            optimizer.zero_grad()
            
            # Forward pass
            reconstruction, mu, logvar = model(x)
            predictions = model.predict(x)
            
            # VAE reconstruction loss
            recon_loss = model.compute_loss(
                (reconstruction, mu, logvar),
                x,  # Reconstruct input
                kld_weight=args.kld_weight
            )
            
            # Supervised prediction loss
            pred_loss = model.compute_supervised_loss(x, y_norm)
            
            # Combined loss
            total_loss = (1 - model.supervised_weight) * recon_loss + \
                        model.supervised_weight * pred_loss
            
            total_loss.backward()
            optimizer.step()
            
            train_recon_loss += recon_loss.item()
            train_pred_loss += pred_loss.item()
            train_total_loss += total_loss.item()
            global_step += 1
            
            if global_step % 50 == 0:
                print(f"[Stage 2] Epoch [{epoch+1}/{args.stage2_epochs}], "
                      f"Step [{global_step}], "
                      f"Total: {total_loss.item():.6f}, "
                      f"Recon: {recon_loss.item():.6f}, "
                      f"Pred: {pred_loss.item():.6f}")
        
        avg_train_recon = train_recon_loss / len(train_loader)
        avg_train_pred = train_pred_loss / len(train_loader)
        avg_train_total = train_total_loss / len(train_loader)
        
        writer.add_scalar('Stage2/train_recon_loss', avg_train_recon, epoch)
        writer.add_scalar('Stage2/train_pred_loss', avg_train_pred, epoch)
        writer.add_scalar('Stage2/train_total_loss', avg_train_total, epoch)
        
        # Validation
        model.eval()
        test_recon_loss = 0
        test_pred_loss = 0
        test_total_loss = 0
        
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                
                y_flat = y.view(y.size(0), -1).double()
                y_norm = (y_flat - y_mean) / y_std
                
                reconstruction, mu, logvar = model(x)
                predictions = model.predict(x)
                
                recon_loss = model.compute_loss(
                    (reconstruction, mu, logvar),
                    x,
                    kld_weight=args.kld_weight
                )
                
                pred_loss = model.compute_supervised_loss(x, y_norm)
                
                total_loss = (1 - model.supervised_weight) * recon_loss + \
                            model.supervised_weight * pred_loss
                
                test_recon_loss += recon_loss.item()
                test_pred_loss += pred_loss.item()
                test_total_loss += total_loss.item()
        
        avg_test_recon = test_recon_loss / len(test_loader)
        avg_test_pred = test_pred_loss / len(test_loader)
        avg_test_total = test_total_loss / len(test_loader)
        
        writer.add_scalar('Stage2/test_recon_loss', avg_test_recon, epoch)
        writer.add_scalar('Stage2/test_pred_loss', avg_test_pred, epoch)
        writer.add_scalar('Stage2/test_total_loss', avg_test_total, epoch)
        
        if avg_test_total < best_test_loss:
            best_test_loss = avg_test_total
            # Save best model
            best_path = os.path.join(save_dir, "best.pth")
            model.save_checkpoint(best_path, optimizer, epoch, avg_test_total)
        
        if (epoch + 1) % 50 == 0:
            print(f"[Stage 2] Epoch {epoch+1}: "
                  f"Train Total = {avg_train_total:.6f} "
                  f"(Recon: {avg_train_recon:.6f}, Pred: {avg_train_pred:.6f}), "
                  f"Test Total = {avg_test_total:.6f}")
    
    print(f"\n[Stage 2] Complete! Best test loss: {best_test_loss:.6f}\n")
    return model


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")
    
    # Create run ID and save directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"semi_supervised_{timestamp}_{args.latent_dim}dim"
    save_dir = f"checkpoints/semi_supervised_vae/{run_id}"
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Run ID: {run_id}")
    print(f"Save directory: {save_dir}\n")
    
    # TensorBoard writer
    writer = SummaryWriter(f"runs/semi_supervised_vae_{timestamp}")
    
    # Load datasets
    unlabeled_dataset, labeled_dataset = load_datasets(args)
    
    # Prepare dataloaders
    (unsupervised_train_loader, supervised_train_loader, 
     supervised_test_loader, test_recon_loader, labeled_train) = prepare_dataloaders(
        unlabeled_dataset, labeled_dataset, args
    )
    
    # Compute target normalization statistics
    y_mean, y_std = compute_target_statistics(supervised_train_loader, device)
    
    # Initialize model
    model = SemiSupervisedVAE(
        compressed_dim=args.latent_dim,
        dropout_prob=args.dropout,
        num_targets=25  # 5 metrics × 5 statistics
    )
    model.double().to(device)
    
    print(f"\nModel: Semi-Supervised VAE")
    print(f"Latent dimension: {args.latent_dim}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}\n")
    
    # Stage 1: Unsupervised pretraining
    model = train_stage1_unsupervised(
        model, unsupervised_train_loader, test_recon_loader,
        args, device, writer
    )
    
    # Save stage 1 checkpoint
    stage1_path = os.path.join(save_dir, "stage1_pretrained.pth")
    model.save_checkpoint(stage1_path, None, args.stage1_epochs, 0)
    print(f"Stage 1 model saved to: {stage1_path}")
    
    # Visualize reconstructions after stage 1
    visualize_reconstructions(model, test_recon_loader, device, 
                             save_dir, num_samples=10)
    
    # Stage 2: Supervised fine-tuning
    model = train_stage2_supervised(
        model, supervised_train_loader, supervised_test_loader,
        y_mean, y_std, args, device, writer, save_dir
    )
    
    # Save final model
    final_path = os.path.join(save_dir, "final.pth")
    model.save_checkpoint(final_path, None, args.stage2_epochs, 0)
    print(f"Final model saved to: {final_path}")
    
    # Save normalization parameters
    scaler_path = os.path.join(save_dir, "scaler.pt")
    torch.save({'mean': y_mean, 'std': y_std}, scaler_path)
    print(f"Scaler saved to: {scaler_path}")
    
    # Visualize reconstructions after stage 2
    visualize_reconstructions(model, test_recon_loader, device,
                             save_dir, num_samples=10)
    
    writer.close()
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print(f"Checkpoints saved in: {save_dir}")
    print(f"TensorBoard logs: runs/semi_supervised_vae_{timestamp}")
    print("\nNext steps:")
    print(f"  1. Evaluate with: python scripts/evaluate_semi_supervised.py --checkpoint {final_path}")
    print(f"  2. View logs: tensorboard --logdir runs/semi_supervised_vae_{timestamp}")
    
    return final_path, run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train semi-supervised VAE with two-stage approach"
    )
    
    # Model parameters
    parser.add_argument("--latent_dim", type=int, default=16,
                       help="Latent space dimension (default: 16)")
    parser.add_argument("--dropout", type=float, default=0.2,
                       help="Dropout probability (default: 0.2)")
    
    # Training parameters
    parser.add_argument("--stage1_epochs", type=int, default=300,
                       help="Epochs for unsupervised pretraining (default: 300)")
    parser.add_argument("--stage2_epochs", type=int, default=200,
                       help="Epochs for supervised fine-tuning (default: 200)")
    parser.add_argument("--batch_size", type=int, default=16,
                       help="Batch size (default: 16)")
    parser.add_argument("--lr", type=float, default=1e-3,
                       help="Learning rate for stage 1 (default: 1e-3)")
    parser.add_argument("--kld_weight", type=float, default=0.00025,
                       help="KL divergence weight (default: 0.00025)")
    parser.add_argument("--supervised_weight", type=float, default=0.3,
                       help="Weight for supervised loss in stage 2 (default: 0.3)")
    
    # Data augmentation
    parser.add_argument("--use_augmentation", action="store_true", default=True,
                       help="Use data augmentation (default: True)")
    parser.add_argument("--augmentation_factor", type=int, default=15,
                       help="Augmentation factor (default: 15)")
    parser.add_argument("--noise_std", type=float, default=0.02,
                       help="Gaussian noise std for augmentation (default: 0.02)")
    parser.add_argument("--savgol_window", type=int, default=15,
                       help="Savitzky-Golay window size (default: 15)")
    parser.add_argument("--savgol_poly", type=int, default=3,
                       help="Savitzky-Golay polynomial order (default: 3)")
    
    args = parser.parse_args()
    
    main(args)
