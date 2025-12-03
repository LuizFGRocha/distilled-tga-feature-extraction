import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import os
import sys
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.augmented_dataset import AugmentedTGADataset
from models.factory import get_model
from src.dataset import TGADataset
from src.visualization import visualize_reconstructions


def save_checkpoint_with_losses(model, path, optimizer, epoch, train_loss, test_loss):
    """Save checkpoint with both train and test loss information."""
    state = {
        'model_state_dict': model.state_dict(),
        'config': model.config if hasattr(model, 'config') else {},
        'epoch': epoch,
        'train_loss': train_loss,
        'test_loss': test_loss,
        'loss': test_loss,  # Keep for backward compatibility
    }
    if optimizer:
        state['optimizer_state_dict'] = optimizer.state_dict()
    torch.save(state, path)


def train(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training {args.model_name} on {device} for task: {args.task}")

    if args.task == 'supervised':
        dataset_mode = 'feature'
    else:
        dataset_mode = 'reconstruction'
    
    original_dataset = TGADataset(data_path=args.data_path, mode=dataset_mode)
    
    train_size = int(0.9 * len(original_dataset))
    test_size = len(original_dataset) - train_size
    train_ds_original, test_ds = random_split(original_dataset, [train_size, test_size])
    
    if args.use_augmentation:
        print(f"Augmenting training set with factor={args.augmentation_factor}...")
        
        train_indices = train_ds_original.indices
        
        train_ds = AugmentedTGADataset(
            data_path=args.data_path,
            noise_std=args.noise_std,
            savgol_window=args.savgol_window,
            savgol_poly=args.savgol_poly,
            augmentation_factor=args.augmentation_factor,
            sample_indices=train_indices
        )
        
        print(f"Training set: {len(train_ds)} samples (augmented)")
        print(f"Test set: {len(test_ds)} samples (original, no augmentation)")
    else:
        train_ds = train_ds_original
    
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size)

    y_mean = None
    y_std = None
    if args.task == 'supervised':
        print("Computing normalization statistics from training set...")
        all_targets = []
        for _, y in train_dl:
            all_targets.append(y)
        
        all_targets = torch.cat(all_targets, dim=0)
        all_targets = all_targets.view(all_targets.size(0), -1).double()
        
        y_mean = all_targets.mean(dim=0).to(device)
        y_std = all_targets.std(dim=0).to(device)
        
        y_std[y_std < 1e-8] = 1.0
        
    model = get_model(args.model_name, compressed_dim=args.latent_dim)
    model.double().to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    writer = SummaryWriter(f"runs/{args.model_name}_exp")
    
    os.makedirs(f"checkpoints/{args.model_name}", exist_ok=True)

    run_id = getattr(args, 'run_id', 'default')
    save_dir = f"checkpoints/{args.model_name}/{run_id}"
    os.makedirs(save_dir, exist_ok=True)
    
    supervised_criterion = nn.MSELoss()
    
    final_path = ""
    best_test_loss = float('inf')
    best_epoch = 0

    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        for batch in train_dl:
            x, target = batch
            x = x.to(device)
            target = target.to(device)
            
            optimizer.zero_grad()
            
            if args.task == 'supervised':
                target = target.view(target.size(0), -1).double()
                
                target = (target - y_mean) / y_std

                output = model(x)
                loss = supervised_criterion(output, target)
            else:
                output = model(x)
                loss = model.compute_loss(output, target, kld_weight=args.kld_weight)

            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            global_step += 1

            if global_step % 100 == 0:
                print(f"Epoch [{epoch+1}/{args.epochs}], Step [{global_step}], Loss: {loss.item():.6f}")

        writer.add_scalar('Loss/train', train_loss / len(train_dl), epoch)

        test_loss = 0
        for batch in test_dl:
            x, target = batch
            x = x.to(device)
            target = target.to(device)
            
            if args.task == 'supervised':
                target = target.view(target.size(0), -1).double()
                
                target = (target - y_mean) / y_std

                output = model(x)
                loss = supervised_criterion(output, target)
            else:
                output = model(x)
                loss = model.compute_loss(output, target, kld_weight=args.kld_weight)
            
            test_loss += loss.item()
            
        avg_test_loss = test_loss / len(test_dl)
        avg_train_loss = train_loss / len(train_dl)
        writer.add_scalar('Loss/test', avg_test_loss, epoch)
        
        # Save best model based on test loss
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            best_epoch = epoch
            best_path = os.path.join(save_dir, "best.pth")
            # Save with both train and test loss
            save_checkpoint_with_losses(model, best_path, optimizer, epoch, 
                                        avg_train_loss, avg_test_loss)
        
        if epoch == args.epochs - 1: # Save at end
            final_path = os.path.join(save_dir, "final.pth")
            # Save with both train and test loss
            save_checkpoint_with_losses(model, final_path, optimizer, epoch,
                                        avg_train_loss, avg_test_loss)
            
            if args.task == 'supervised':
                scaler_path = os.path.join(save_dir, "scaler.pt")
                torch.save({'mean': y_mean, 'std': y_std}, scaler_path)
            
            if args.task == 'reconstruction':
                visualize_reconstructions(model, test_dl, device, save_dir, num_samples=15)
            
    return final_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--data_path", type=str, default="./data/tga/data.npz")
    parser.add_argument("--epochs", type=int, default=1500)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--latent_dim", type=int, default=64)
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument("--kld_weight", type=float, default=0.00025, help="Weight for KL Divergence loss in VAE")
    parser.add_argument('--task', type=str, default='reconstruction', choices=['reconstruction', 'supervised'], help='Task type: reconstruction (unsupervised) or supervised (regression)')
    parser.add_argument("--use_augmentation", default=True, help="Use augmented dataset")
    parser.add_argument("--noise_std", type=float, default=0.02, help="Std dev of Gaussian noise for augmentation")
    parser.add_argument("--savgol_window", type=int, default=15, help="Savitzky-Golay window size (must be odd)")
    parser.add_argument("--savgol_poly", type=int, default=3, help="Savitzky-Golay polynomial order")
    parser.add_argument("--augmentation_factor", type=int, default=15, help="Number of augmented versions per original sample (e.g., 5 = 5x more data)")
    
    args = parser.parse_args()
    train(args)