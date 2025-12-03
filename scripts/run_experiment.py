import argparse
import pandas as pd
import os
import datetime
import json
import sys

# Add project root to path so we can import from scripts/
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.train import train
from scripts.evaluate import run_evaluation # We will wrap the main logic of evaluate.py into a function

LEADERBOARD_PATH = "experiments_leaderboard.csv"

def main():
    parser = argparse.ArgumentParser(description="Run a full TGA experiment: Train -> Eval -> Log")
    
    parser.add_argument("--model_name", type=str, required=True, help="Name of model in factory")
    parser.add_argument("--latent_dim", type=int, default=64)
    parser.add_argument("--note", type=str, default="", help="Short description of this experiment")
    
    parser.add_argument("--epochs", type=int, default=1500)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--kld_weight", type=float, default=0.00025, help="Weight for KL Divergence loss in VAE")
    
    args = parser.parse_args()

    # Generate a unique Run ID
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{args.model_name}_{timestamp}_{args.latent_dim}dim"
    
    print(f"=== Starting Experiment: {run_id} ===")
    
    print("\n[1/3] Starting Training...")
    
    train_args = argparse.Namespace(
        model_name=args.model_name,
        data_path="./data/tga/data.npz",
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        latent_dim=args.latent_dim,
        save_interval=500,
        run_id=run_id, # Pass run_id to save checkpoint specifically for this run
        kld_weight=args.kld_weight
    )
    
    checkpoint_path = train(train_args) 
    
    print(f"Training finished. Checkpoint saved at: {checkpoint_path}")
    print("\n[2/3] Starting Evaluation...")
    
    eval_args = argparse.Namespace(
        model_name=args.model_name,
        checkpoint_path=checkpoint_path,
        data_path="./data/tga_afm/data.npz",
        latent_dim=args.latent_dim,
        method='bootstrap'
    )
    
    eval_results = run_evaluation(eval_args)
    
    print("\n[3/3] Logging Results...")
    
    log_entry = {
        "timestamp": timestamp,
        "run_id": run_id,
        "model": args.model_name,
        "latent_dim": args.latent_dim,
        "epochs": args.epochs,
        "lr": args.lr,
        "note": args.note,
        "checkpoint": checkpoint_path
    }
    
    # Add metrics dynamically
    for res in eval_results:
        metric_name = res['Target Property']
        log_entry[f"{metric_name} (R2)"] = round(res['Mean R^2'], 4)
        log_entry[f"{metric_name} (Low CI)"] = round(res['95% CI Lower'], 4)
        log_entry[f"{metric_name} (High CI)"] = round(res['95% CI Upper'], 4)

    # Save to CSV
    df = pd.DataFrame([log_entry])
    
    if not os.path.exists(LEADERBOARD_PATH):
        df.to_csv(LEADERBOARD_PATH, index=False)
    else:
        df.to_csv(LEADERBOARD_PATH, mode='a', header=False, index=False)
        
    print(f"Experiment logged to {LEADERBOARD_PATH}")
    print("=== Done ===")

if __name__ == "__main__":
    main()