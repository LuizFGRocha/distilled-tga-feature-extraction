import matplotlib.pyplot as plt
import torch
import os

def visualize_reconstructions(model, test_dl, device, save_dir, num_samples=None):
    """
    Visualiza reconstruções do modelo para amostras do conjunto de teste.
    Se num_samples=None, plota todas as amostras do conjunto de teste.
    """
    model.eval()
    
    all_x = []
    all_targets = []
    
    with torch.no_grad():
        for x, target in test_dl:
            x = x.to(device)
            all_x.append(x)
            all_targets.append(target)
        
        x = torch.cat(all_x, dim=0)
        
        output = model(x)
        
        if isinstance(output, tuple):
            recon_x = output[0]
        else:
            recon_x = output
    
    x_np = x.cpu().numpy()
    recon_np = recon_x.cpu().numpy()
    
    if num_samples is None:
        num_samples = x_np.shape[0]
    else:
        num_samples = min(num_samples, x_np.shape[0])
    
    fig, axes = plt.subplots(num_samples, 2, figsize=(12, 4 * num_samples))
    
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_samples):
        axes[i, 0].plot(x_np[i, 0], label='Original', color='blue', alpha=0.7, linewidth=1.5)
        axes[i, 0].plot(recon_np[i, 0], label='Reconstruído', color='red', linestyle='--', alpha=0.7, linewidth=1.5)
        axes[i, 0].set_title(f'Amostra {i+1} - Weight (W)')
        axes[i, 0].set_xlabel('Pontos')
        axes[i, 0].set_ylabel('W')
        axes[i, 0].legend()
        axes[i, 0].grid(True, alpha=0.3)
        
        axes[i, 1].plot(x_np[i, 1], label='Original', color='blue', alpha=0.7, linewidth=1.5)
        axes[i, 1].plot(recon_np[i, 1], label='Reconstruído', color='red', linestyle='--', alpha=0.7, linewidth=1.5)
        axes[i, 1].set_title(f'Amostra {i+1} - Derivative (dW/dT)')
        axes[i, 1].set_xlabel('Pontos')
        axes[i, 1].set_ylabel('dW/dT')
        axes[i, 1].legend()
        axes[i, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, 'reconstructions.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Visualização de reconstruções salva em: {save_path}")
    plt.close()
