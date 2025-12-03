# TGA Feature Extraction

A repository to unify the efforts to improve graphene TGA feature extraction in the nanocomputing projects of 2025/2.

## Quick Start

### 1. Run a Full Experiment
To train a model, evaluate it, and log the results to the leaderboard in one go:

```bash
python scripts/run_experiment.py --model_name attention_unet --note "Baseline run"
```

This command will:
1. Train the `attention_unet` on the TGA data.
2. Save the best checkpoint to `checkpoints/attention_unet/<run_id>/`.
3. Evaluate the encodings on the AFM regression task.
4. Append the results to `experiments_leaderboard.csv`.

### 2. Check Results
View the leaderboard to compare models:
```bash
cat experiments_leaderboard.csv
```

---

## Detailed Usage

### Training Only
If you only want to train a model without evaluating:

```bash
python scripts/train.py --model_name attention_unet --epochs 1500 --batch_size 16
```

### Evaluation Only
If you have a saved checkpoint and want to check its performance:

```bash
python scripts/evaluate.py \
  --model_name attention_unet \
  --checkpoint_path checkpoints/attention_unet/<run_id>/final.pth \
  --method cv
```
* `method`: Choose between `bootstrap` (default) or `cv` (Cross-Validation).

---

## Adding a New Model

1. **Create the Model File**:
   Create a new file in `models/`, e.g., `models/my_new_model.py`.
   Inherit from `TGAFeatureExtractor` and implement `forward` and `encode`.

   ```python
   from models.base import TGAFeatureExtractor
   import torch.nn as nn

   class MyNewModel(TGAFeatureExtractor):
       def __init__(self, compressed_dim=64):
           super().__init__()
           self.encoder = nn.Linear(1024, compressed_dim)
           # ... define layers ...

       def encode(self, x):
           # Return the latent vector
           return self.encoder(x)
       
       def forward(self, x):
           # Return reconstruction
           return x 
   ```

2. **Register the Model**:
   Open `models/factory.py` and add your model to the `get_model` function.

   ```python
   from models.my_new_model import MyNewModel

   def get_model(model_name, **kwargs):
       # ...
       elif model_name == 'my_new_model':
           return MyNewModel(**kwargs)
   ```

3. **Run**:
   ```bash
   python scripts/run_experiment.py --model_name my_new_model --note "Testing new architecture"
   ```

## 📓 Notebooks
When working in `notebooks/`, add this snippet to import project modules:

```python
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

from models.factory import get_model
```