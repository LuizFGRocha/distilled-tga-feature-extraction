"""
Evaluate Semi-Supervised VAE

Tests the learned representations using Random Forest regression on AFM targets.
Compares performance with the unsupervised baseline.

Usage:
    python scripts/evaluate_semi_supervised.py --checkpoint checkpoints/semi_supervised_vae/.../final.pth
"""

import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.semi_supervised_vae import SemiSupervisedVAE
from src.dataset import TGADataset
from torch.utils.data import DataLoader
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def load_model(checkpoint_path, latent_dim, device):
    """Load trained semi-supervised VAE"""
    model = SemiSupervisedVAE(compressed_dim=latent_dim, num_targets=25)
    model.load_checkpoint(checkpoint_path, device)
    model.to(device)
    model.double()
    model.eval()
    return model


def generate_encodings(model, data_loader, device):
    """Generate latent encodings for all samples"""
    encodings = []
    all_labels = []
    
    with torch.no_grad():
        for x_batch, y_batch in data_loader:
            x_batch = x_batch.to(device)
            encoding = model.encode(x_batch).cpu().numpy()
            encodings.append(encoding)
            all_labels.append(y_batch.numpy())
    
    encodings = np.vstack(encodings)
    Y = np.vstack(all_labels)
    
    return encodings, Y


def evaluate_with_random_forest_loo(encodings, labels, n_estimators=100, 
                                     max_depth=10, random_state=42):
    """
    Evaluate using Random Forest with Leave-One-Out Cross-Validation.
    """
    loo = LeaveOneOut()
    predictions = np.zeros_like(labels)
    
    for train_idx, test_idx in loo.split(encodings):
        X_train, X_test = encodings[train_idx], encodings[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]
        
        rf = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1
        )
        rf.fit(X_train, y_train)
        
        pred = rf.predict(X_test)
        predictions[test_idx] = pred
    
    r2 = r2_score(labels, predictions)
    mse = mean_squared_error(labels, predictions)
    mae = mean_absolute_error(labels, predictions)
    
    return r2, mse, mae, predictions


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")
    
    # Load dataset
    dataset = TGADataset(data_path='./data/tga_afm/data.npz', mode='feature')
    dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    
    print(f"Dataset size: {len(dataset)} samples")
    
    # Load model
    print(f"Loading model from: {args.checkpoint}")
    model = load_model(args.checkpoint, args.latent_dim, device)
    print(f"Model loaded successfully")
    print(f"Latent dimension: {args.latent_dim}\n")
    
    # Generate encodings
    print("Generating encodings...")
    encodings, Y = generate_encodings(model, dataloader, device)
    print(f"Encodings shape: {encodings.shape}")
    print(f"Labels shape: {Y.shape}\n")
    
    # Check for NaN/Inf
    print("Checking encoding quality...")
    print(f"  NaN in encodings: {np.isnan(encodings).any()}")
    print(f"  Inf in encodings: {np.isinf(encodings).any()}")
    print(f"  Encoding stats - Min: {encodings.min():.4f}, "
          f"Max: {encodings.max():.4f}, Mean: {encodings.mean():.4f}\n")
    
    # Standardize encodings
    scaler = StandardScaler()
    encodings_scaled = scaler.fit_transform(encodings)
    
    # Define target configurations
    stat_names = ['Mean', 'Variance', 'Skewness', 'Kurtosis', 'Median']
    metric_names = ['Min Ferret', 'Max Ferret', 'Height', 'Area', 'Volume']
    
    label_configs = []
    for i, metric in enumerate(metric_names):
        for j, stat in enumerate(stat_names):
            label_configs.append({
                'name': f'{metric} {stat}',
                'data': Y[:, i, j]
            })
    
    print(f"Testing {len(label_configs)} target variables")
    
    # Run Random Forest evaluation
    print("="*80)
    print("RANDOM FOREST REGRESSION RESULTS (LOO CV)")
    print("="*80)
    
    results = []
    
    for config in label_configs:
        labels = config['data']
        
        print(f"\nEvaluating: {config['name']}")
        
        r2, mse, mae, predictions = evaluate_with_random_forest_loo(
            encodings_scaled,
            labels,
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            random_state=42
        )
        
        results.append({
            'Target': config['name'],
            'R² Score': r2,
            'MSE': mse,
            'MAE': mae,
            'Predictions': predictions
        })
        
        print(f"  R² = {r2:.4f}")
        print(f"  MSE = {mse:.4f}")
        print(f"  MAE = {mae:.4f}")
    
    print("\n" + "="*80)
    
    # Create results dataframe
    results_df = pd.DataFrame([{
        'Target': r['Target'],
        'R² Score': r['R² Score'],
        'MSE': r['MSE'],
        'MAE': r['MAE']
    } for r in results])
    
    results_df = results_df.sort_values('R² Score', ascending=False)
    
    print("\nSUMMARY TABLE")
    print("="*80)
    print(results_df.to_string(index=False))
    
    avg_r2 = results_df['R² Score'].mean()
    print(f"\n{'='*80}")
    print(f"Average R² Score: {avg_r2:.4f}")
    print(f"Best Target: {results_df.iloc[0]['Target']} (R² = {results_df.iloc[0]['R² Score']:.4f})")
    print(f"Worst Target: {results_df.iloc[-1]['Target']} (R² = {results_df.iloc[-1]['R² Score']:.4f})")
    print(f"{'='*80}\n")
    
    # Save results
    output_dir = os.path.dirname(args.checkpoint)
    results_path = os.path.join(output_dir, "evaluation_results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"Results saved to: {results_path}\n")
    
    # Visualize results
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = ['green' if r2 > 0 else 'red' for r2 in results_df['R² Score']]
    bars = ax.barh(results_df['Target'], results_df['R² Score'], 
                   color=colors, alpha=0.7, edgecolor='black')
    
    ax.set_xlabel('R² Score', fontsize=12, fontweight='bold')
    ax.set_ylabel('Target Variable', fontsize=12, fontweight='bold')
    ax.set_title(f'Semi-Supervised VAE Performance (latent_dim={args.latent_dim})',
                 fontsize=14, fontweight='bold')
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax.grid(True, alpha=0.3, axis='x')
    
    for bar, r2 in zip(bars, results_df['R² Score']):
        width = bar.get_width()
        label_x_pos = width + 0.02 if width >= 0 else width - 0.02
        ha = 'left' if width >= 0 else 'right'
        ax.text(label_x_pos, bar.get_y() + bar.get_height()/2, f'{r2:.3f}',
                ha=ha, va='center', fontweight='bold', fontsize=9)
    
    plt.tight_layout()
    fig_path = os.path.join(output_dir, "evaluation_r2_scores.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {fig_path}")
    plt.close()
    
    # Create prediction scatter plots for top 4 targets
    top_4_results = results_df.head(4)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.ravel()
    
    for idx, (_, row) in enumerate(top_4_results.iterrows()):
        target_name = row['Target']
        r2 = row['R² Score']
        
        result = [r for r in results if r['Target'] == target_name][0]
        predictions = result['Predictions']
        actual = [config['data'] for config in label_configs 
                 if config['name'] == target_name][0]
        
        axes[idx].scatter(actual, predictions, alpha=0.6, s=100, 
                         edgecolors='black', linewidth=1)
        
        min_val = min(actual.min(), predictions.min())
        max_val = max(actual.max(), predictions.max())
        axes[idx].plot([min_val, max_val], [min_val, max_val], 
                      'r--', linewidth=2, label='Perfect Prediction')
        
        axes[idx].set_xlabel('Actual Value', fontsize=11, fontweight='bold')
        axes[idx].set_ylabel('Predicted Value', fontsize=11, fontweight='bold')
        axes[idx].set_title(f'{target_name}\nR² = {r2:.4f}', 
                           fontsize=12, fontweight='bold')
        axes[idx].legend(loc='best')
        axes[idx].grid(True, alpha=0.3)
    
    plt.suptitle('Top 4 Predictions: Semi-Supervised VAE', 
                 fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    fig_path = os.path.join(output_dir, "evaluation_top4_predictions.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {fig_path}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate semi-supervised VAE using Random Forest"
    )
    
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--latent_dim", type=int, default=16,
                       help="Latent space dimension (default: 16)")
    parser.add_argument("--n_estimators", type=int, default=100,
                       help="Number of trees in Random Forest (default: 100)")
    parser.add_argument("--max_depth", type=int, default=10,
                       help="Max depth of trees (default: 10)")
    
    args = parser.parse_args()
    
    main(args)
