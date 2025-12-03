import numpy as np
import torch
from models.autoencoder import ConvAutoencoder


# Carregue o dataset
data = np.load('./data/sets/afm_stats/data.npz')
X = data['X']
Y = data['Y']
samples = data['samples']

# Carregue o seu modelo
model = ConvAutoencoder()
model.load_state_dict(torch.load('checkpoints/unet/14500.pth', weights_only=True))
model.to('cpu')
model.double()

# Selecione quais curvas do TGA seu modelo usa no encoding. Ex.: w, dwdt
X = X[:, 1:3, :]

# Transforme cada entrada com o modelo:
for i in range(X.shape[0]):
    t = torch.tensor(X[i])
    X[i] = model.encode(t).detach().numpy()

# Salve o dataset atualizado com os vetores de features
np.savez_compressed('data.npz', X=X, Y=Y, samples=samples)