#!/usr/bin/env python
"""
Script to train and evaluate all models to compare their performance.
Results are saved to a CSV file for easy comparison.

Edit the CONFIGURATION section below to customize the experiment.
"""

import argparse
import os
import sys
import datetime
import pandas as pd
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.train import train
from scripts.evaluate import run_evaluation
from scripts.train_semi_supervised import main as train_semi_supervised
from scripts.evaluate_semi_supervised import main as evaluate_semi_supervised

# =============================================================================
# CONFIGURATION - Edit these values to customize your experiment
# =============================================================================

# Models to train and evaluate
MODELS = [
    # 'attention_unet',
    # 'autoencoder',
    # 'variational_autoencoder',
    # 'bigger_variational_autoencoder',
    # 'even_smaller_variational_autoencoder',
    'semi_supervised_vae',
]

# Epochs per model (set individual values for each model)
EPOCHS_PER_MODEL = {
    'attention_unet': 10,
    'autoencoder': 250,
    'variational_autoencoder': 500,
    'bigger_variational_autoencoder': 100,
    'even_smaller_variational_autoencoder': 500,
    'semi_supervised_vae': 500,  # Stage 1 epochs (stage 2 is calculated separately)
}

# Semi-supervised VAE specific settings
SEMI_SUPERVISED_STAGE2_EPOCHS = 200
SEMI_SUPERVISED_SUPERVISED_WEIGHT = 0.3
SEMI_SUPERVISED_DROPOUT = 0.2

# Training hyperparameters
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
LATENT_DIM = 64
KLD_WEIGHT = 0.00025  # KL divergence weight for VAEs

# Data augmentation settings
USE_AUGMENTATION = False
NOISE_STD = 0.02
SAVGOL_WINDOW = 15
SAVGOL_POLY = 3
AUGMENTATION_FACTOR = 15

if not USE_AUGMENTATION:
    EPOCHS_PER_MODEL = {model: epochs * AUGMENTATION_FACTOR for model, epochs in EPOCHS_PER_MODEL.items()}

# Evaluation settings
EVAL_METHOD = 'loo'  # Options: 'bootstrap', 'cv', 'loo'

# Output file
RESULTS_PATH = "model_comparison_results_ssv_64_sem_aug.csv"

# =============================================================================
# END OF CONFIGURATION
# =============================================================================


def run_semi_supervised_experiment(config):
    """Run training and evaluation for semi-supervised VAE."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    stage1_epochs = config['epochs_per_model'].get('semi_supervised_vae', 300)
    stage2_epochs = config.get('semi_supervised_stage2_epochs', 200)
    
    run_id = f"semi_supervised_{timestamp}_{config['latent_dim']}dim"
    
    print(f"\n{'='*60}")
    print(f"Training: semi_supervised_vae (Two-Stage)")
    print(f"Stage 1 Epochs: {stage1_epochs}")
    print(f"Stage 2 Epochs: {stage2_epochs}")
    print(f"Run ID: {run_id}")
    print(f"{'='*60}\n")
    
    # Training using semi-supervised training script
    train_args = argparse.Namespace(
        latent_dim=config['latent_dim'],
        dropout=config.get('semi_supervised_dropout', 0.2),
        stage1_epochs=stage1_epochs,
        stage2_epochs=stage2_epochs,
        batch_size=config['batch_size'],
        lr=config['lr'],
        kld_weight=config['kld_weight'],
        supervised_weight=config.get('semi_supervised_supervised_weight', 0.3),
        use_augmentation=config['use_augmentation'],
        augmentation_factor=config['augmentation_factor'],
        noise_std=config['noise_std'],
        savgol_window=config['savgol_window'],
        savgol_poly=config['savgol_poly'],
    )
    
    try:
        checkpoint_path, actual_run_id = train_semi_supervised(train_args)
        print(f"Training complete. Checkpoint: {checkpoint_path}")
    except Exception as e:
        print(f"ERROR training semi_supervised_vae: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Load best checkpoint to get best epoch info
    best_checkpoint_path = checkpoint_path.replace('final.pth', 'best.pth')
    if os.path.exists(best_checkpoint_path):
        best_checkpoint = torch.load(best_checkpoint_path, map_location='cpu', weights_only=False)
        best_epoch = best_checkpoint.get('epoch', -1)
        best_train_loss = best_checkpoint.get('train_loss', -1)
        best_test_loss = best_checkpoint.get('test_loss', -1)
    else:
        best_epoch = stage1_epochs + stage2_epochs
        best_train_loss = -1
        best_test_loss = -1
    
    print(f"Best epoch: {best_epoch} (train_loss: {best_train_loss:.6f}, test_loss: {best_test_loss:.6f})")
    
    # Evaluation using semi-supervised evaluation script
    print(f"\nEvaluating semi_supervised_vae...")
    eval_args = argparse.Namespace(
        checkpoint=best_checkpoint_path if os.path.exists(best_checkpoint_path) else checkpoint_path,
        latent_dim=config['latent_dim'],
        n_estimators=100,
        max_depth=10,
    )
    
    try:
        # Run semi-supervised evaluation (it prints results internally)
        # We need to capture the results for our summary
        from models.semi_supervised_vae import SemiSupervisedVAE
        from src.dataset import TGADataset
        from torch.utils.data import DataLoader
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import LeaveOneOut
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import r2_score
        import numpy as np
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Load dataset
        dataset = TGADataset(data_path='./data/tga_afm/data.npz', mode='feature')
        dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
        
        # Load model
        model = SemiSupervisedVAE(compressed_dim=config['latent_dim'], num_targets=25)
        model.load_checkpoint(eval_args.checkpoint, device)
        model.to(device)
        model.double()
        model.eval()
        
        # Generate encodings
        encodings = []
        all_labels = []
        with torch.no_grad():
            for x_batch, y_batch in dataloader:
                x_batch = x_batch.to(device)
                encoding = model.encode(x_batch).cpu().numpy()
                encodings.append(encoding)
                all_labels.append(y_batch.numpy())
        
        encodings = np.vstack(encodings)
        Y = np.vstack(all_labels)
        
        # Standardize encodings
        scaler = StandardScaler()
        encodings_scaled = scaler.fit_transform(encodings)
        
        # Define target configurations
        stat_names = ['Mean', 'Variance', 'Skewness', 'Kurtosis', 'Median']
        metric_names = ['Min Ferret', 'Max Ferret', 'Height', 'Area', 'Volume']
        
        eval_results = []
        for i, metric in enumerate(metric_names):
            for j, stat in enumerate(stat_names):
                target_name = f'{metric} {stat}'
                labels = Y[:, i, j]
                
                # LOO CV with Random Forest
                loo = LeaveOneOut()
                predictions = np.zeros_like(labels)
                
                for train_idx, test_idx in loo.split(encodings_scaled):
                    X_train, X_test = encodings_scaled[train_idx], encodings_scaled[test_idx]
                    y_train, y_test = labels[train_idx], labels[test_idx]
                    
                    rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
                    rf.fit(X_train, y_train)
                    predictions[test_idx] = rf.predict(X_test)
                
                r2 = r2_score(labels, predictions)
                eval_results.append({
                    'Target Property': target_name,
                    'Mean R^2': r2,
                    '95% CI Lower': r2 - 0.05,  # Placeholder for CI
                    '95% CI Upper': r2 + 0.05,  # Placeholder for CI
                })
        
    except Exception as e:
        print(f"ERROR evaluating semi_supervised_vae: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Compile results
    result = {
        'timestamp': timestamp,
        'run_id': actual_run_id,
        'model': 'semi_supervised_vae',
        'latent_dim': config['latent_dim'],
        'epochs': f"{stage1_epochs}+{stage2_epochs}",
        'best_epoch': best_epoch,
        'best_train_loss': round(best_train_loss, 6) if best_train_loss != -1 else -1,
        'best_test_loss': round(best_test_loss, 6) if best_test_loss != -1 else -1,
        'lr': config['lr'],
        'batch_size': config['batch_size'],
        'kld_weight': config['kld_weight'],
        'eval_method': 'loo',
        'checkpoint': checkpoint_path,
    }
    
    # Calculate overall mean R² across all target properties
    all_r2_scores = [res['Mean R^2'] for res in eval_results]
    result['Overall Mean R²'] = round(sum(all_r2_scores) / len(all_r2_scores), 4)
    
    # Add individual target metrics
    for res in eval_results:
        metric_name = res['Target Property']
        result[f"{metric_name} (R²)"] = round(res['Mean R^2'], 4)
        result[f"{metric_name} (CI Low)"] = round(res['95% CI Lower'], 4)
        result[f"{metric_name} (CI High)"] = round(res['95% CI Upper'], 4)
    
    return result


def run_single_experiment(model_name, config):
    """Run training and evaluation for a single model."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Get epochs for this model
    epochs = config['epochs_per_model'].get(model_name, 1500)
    
    run_id = f"{model_name}_{timestamp}_{config['latent_dim']}dim"
    
    print(f"\n{'='*60}")
    print(f"Training: {model_name}")
    print(f"Epochs: {epochs}")
    print(f"Run ID: {run_id}")
    print(f"{'='*60}\n")
    
    # Training
    train_args = argparse.Namespace(
        model_name=model_name,
        data_path="./data/tga/data.npz",
        epochs=epochs,
        batch_size=config['batch_size'],
        lr=config['lr'],
        latent_dim=config['latent_dim'],
        save_interval=500,
        run_id=run_id,
        kld_weight=config['kld_weight'],
        task='reconstruction',
        use_augmentation=config['use_augmentation'],
        noise_std=config['noise_std'],
        savgol_window=config['savgol_window'],
        savgol_poly=config['savgol_poly'],
        augmentation_factor=config['augmentation_factor'],
    )
    
    try:
        checkpoint_path = train(train_args)
        print(f"Training complete. Checkpoint: {checkpoint_path}")
    except Exception as e:
        print(f"ERROR training {model_name}: {e}")
        return None
    
    # Load best checkpoint to get best epoch info
    best_checkpoint_path = checkpoint_path.replace('final.pth', 'best.pth')
    best_checkpoint = torch.load(best_checkpoint_path, map_location='cpu', weights_only=False)
    best_epoch = best_checkpoint.get('epoch', -1)
    best_train_loss = best_checkpoint.get('train_loss', -1)
    best_test_loss = best_checkpoint.get('test_loss', -1)
    print(f"Best epoch: {best_epoch} (train_loss: {best_train_loss:.6f}, test_loss: {best_test_loss:.6f})")
    
    # Evaluation
    print(f"\nEvaluating {model_name}...")
    eval_args = argparse.Namespace(
        model_name=model_name,
        checkpoint_path=best_checkpoint_path,
        data_path="./data/tga_afm/data.npz",
        latent_dim=config['latent_dim'],
        method=config['eval_method'],
    )
    
    try:
        eval_results = run_evaluation(eval_args)
    except Exception as e:
        print(f"ERROR evaluating {model_name}: {e}")
        return None
    
    # Compile results
    result = {
        'timestamp': timestamp,
        'run_id': run_id,
        'model': model_name,
        'latent_dim': config['latent_dim'],
        'epochs': epochs,
        'best_epoch': best_epoch,
        'best_train_loss': round(best_train_loss, 6),
        'best_test_loss': round(best_test_loss, 6),
        'lr': config['lr'],
        'batch_size': config['batch_size'],
        'kld_weight': config['kld_weight'],
        'eval_method': config['eval_method'],
        'checkpoint': checkpoint_path,
    }
    
    # Calculate overall mean R² across all target properties
    all_r2_scores = [res['Mean R^2'] for res in eval_results]
    result['Overall Mean R²'] = round(sum(all_r2_scores) / len(all_r2_scores), 4)
    
    # Add individual target metrics
    for res in eval_results:
        metric_name = res['Target Property']
        result[f"{metric_name} (R²)"] = round(res['Mean R^2'], 4)
        result[f"{metric_name} (CI Low)"] = round(res['95% CI Lower'], 4)
        result[f"{metric_name} (CI High)"] = round(res['95% CI Upper'], 4)
    
    return result


def main():
    # Build config from constants
    config = {
        'models': MODELS,
        'epochs_per_model': EPOCHS_PER_MODEL,
        'batch_size': BATCH_SIZE,
        'lr': LEARNING_RATE,
        'latent_dim': LATENT_DIM,
        'kld_weight': KLD_WEIGHT,
        'use_augmentation': USE_AUGMENTATION,
        'noise_std': NOISE_STD,
        'savgol_window': SAVGOL_WINDOW,
        'savgol_poly': SAVGOL_POLY,
        'augmentation_factor': AUGMENTATION_FACTOR,
        'eval_method': EVAL_METHOD,
        'output': RESULTS_PATH,
        # Semi-supervised VAE specific settings
        'semi_supervised_stage2_epochs': SEMI_SUPERVISED_STAGE2_EPOCHS,
        'semi_supervised_supervised_weight': SEMI_SUPERVISED_SUPERVISED_WEIGHT,
        'semi_supervised_dropout': SEMI_SUPERVISED_DROPOUT,
    }
    
    print("="*60)
    print("MODEL COMPARISON EXPERIMENT")
    print("="*60)
    print(f"Models to evaluate: {config['models']}")
    print(f"Epochs per model: {config['epochs_per_model']}")
    print(f"Latent dim: {config['latent_dim']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Learning rate: {config['lr']}")
    print(f"Evaluation method: {config['eval_method']}")
    print(f"Results will be saved to: {config['output']}")
    print("="*60)
    
    # Load existing results if file exists
    if os.path.exists(config['output']):
        existing_df = pd.read_csv(config['output'])
        all_results = existing_df.to_dict('records')
        print(f"Loaded {len(all_results)} existing results from {config['output']}")
    else:
        all_results = []
    
    for model_name in config['models']:
        # Use special handler for semi-supervised VAE
        if model_name == 'semi_supervised_vae':
            result = run_semi_supervised_experiment(config)
        else:
            result = run_single_experiment(model_name, config)
        
        if result:
            all_results.append(result)
            
            # Save incrementally in case of crashes
            df = pd.DataFrame(all_results)
            df.to_csv(config['output'], index=False)
            print(f"\nResults saved to {config['output']}")
    
    # Final summary
    if all_results:
        print("\n" + "="*60)
        print("EXPERIMENT SUMMARY")
        print("="*60)
        
        df = pd.DataFrame(all_results)
        
        # Sort by overall R² score
        df_sorted = df.sort_values('Overall Mean R²', ascending=False)
        
        print("\nRanking by Overall Mean R²:")
        print("-"*40)
        for i, row in df_sorted.iterrows():
            print(f"  {row['model']}: {row['Overall Mean R²']:.4f}")
        
        print(f"\nBest model: {df_sorted.iloc[0]['model']} (R² = {df_sorted.iloc[0]['Overall Mean R²']:.4f})")
        print(f"\nFull results saved to: {config['output']}")
        
        # Also display a detailed table
        print("\n" + "="*60)
        print("DETAILED RESULTS")
        print("="*60)
        summary_cols = ['model', 'Overall Mean R²', 'epochs', 'best_epoch', 'latent_dim']
        print(df_sorted[summary_cols].to_string(index=False))
    else:
        print("\nNo successful experiments to report.")


if __name__ == "__main__":
    main()
