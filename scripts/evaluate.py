import argparse
import os
import sys
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler

# Add project root to path so we can import from scripts/
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from the new modular structure
from models.factory import get_model
from src.dataset import TGADataset
from src.evaluation import evaluate_with_bootstrap, evaluate_with_cv, evaluate_with_loo, visualize_1d_fit

def generate_encodings(model, data_loader, device):
    """
    Runs inference to generate encodings and collects corresponding labels.
    """
    encodings = []
    all_labels = []
    
    model.eval()
    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(device)
            encoding = model.encode(x).cpu().numpy()
            
            encodings.append(encoding)
            all_labels.append(y.numpy())
            
    return np.vstack(encodings), np.vstack(all_labels)

def run_evaluation(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Evaluating on device: {device}")

    dataset = TGADataset(data_path=args.data_path, mode='feature')
    data_loader = DataLoader(dataset, batch_size=32, shuffle=False)
    print(f"Loaded dataset with {len(dataset)} samples.")

    # Load Model
    print(f"Loading {args.model_name} from {args.checkpoint_path}...")
    model = get_model(args.model_name, compressed_dim=args.latent_dim)
    
    model.load_checkpoint(args.checkpoint_path, device)

    model.to(device)
    model.double()

    print("Generating encodings...")
    encodings, Y = generate_encodings(model, data_loader, device)
    
    scaler = StandardScaler()
    encodings_scaled = scaler.fit_transform(encodings)

    # Define Targets
    label_configs = [
        {'name': 'Min Ferret Mean',   'data': Y[:, 0, 0]},
        {'name': 'Max Ferret Mean',   'data': Y[:, 1, 0]},
        {'name': 'Height Mean',       'data': Y[:, 2, 0]},
        {'name': 'Area Mean',         'data': Y[:, 3, 0]},
        {'name': 'Volume Mean',       'data': Y[:, 4, 0]},
        {'name': 'Min Ferret Median', 'data': Y[:, 0, 1]},
        {'name': 'Max Ferret Median', 'data': Y[:, 1, 1]},
        {'name': 'Height Median',     'data': Y[:, 2, 1]},
        {'name': 'Area Median',       'data': Y[:, 3, 1]},
        {'name': 'Volume Median',     'data': Y[:, 4, 1]}
    ]
    
    # Run Evaluation
    results = []
    print(f"\n--- Encoding Quality Evaluation ({args.method}) ---")
    
    for config in label_configs:
        labels = config['data']
        
        if args.method == 'bootstrap':
            mean_r2, lower_ci, upper_ci = evaluate_with_bootstrap(encodings_scaled, labels)
        elif args.method == 'cv':
            mean_r2, lower_ci, upper_ci = evaluate_with_cv(encodings_scaled, labels)
        elif args.method == 'loo':
            mean_r2, lower_ci, upper_ci = evaluate_with_loo(encodings_scaled, labels)

        visualize_1d_fit(encodings_scaled, labels, split_seed=42)
            
        results.append({
            'Target Property': config['name'],
            'Mean R^2': mean_r2,
            '95% CI Lower': lower_ci,
            '95% CI Upper': upper_ci
        })
        
    return results

def main():
    parser = argparse.ArgumentParser(description="Evaluate encoding quality of TGA models.")
    parser.add_argument('--model_name', type=str, required=True, choices=['attention_unet', 'autoencoder', 'variational_autoencoder'], help='Model architecture name.')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to the .pth checkpoint.')
    parser.add_argument('--data_path', type=str, default='./data/tga_afm/data.npz', help='Path to the evaluation dataset (must contain X and Y).')
    parser.add_argument('--latent_dim', type=int, default=64, help='Dimension of the latent space.')
    parser.add_argument('--method', type=str, default='loo', choices=['bootstrap', 'cv', 'loo'], help='Evaluation method.')
    args = parser.parse_args()

    results = run_evaluation(args)
    print(pd.DataFrame(results))
    print("\nEvaluation complete.")

if __name__ == '__main__':
    main()