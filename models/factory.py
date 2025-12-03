from models.attention_unet import AttentionUNet
from models.autoencoder import ConvAutoencoder
from models.variational_autoencoder import VAE
from models.bigger_variational_autoencoder import Bigger_VAE
from models.supervised_cnn import SupervisedCNN
from models.small_variational_autoencoder import SmallVAE
from models.micro_variational_autoencoder import MicroVAE
from models.nano_variational_autoencoder import NanoVAE
from models.even_smaller_variational_autoencoder import EvenSmallerVAE
from models.semi_supervised_vae import SemiSupervisedVAE

def get_model(model_name, **kwargs):
    """
    Factory function to instantiate models.
    
    Args:
        model_name (str): Name of the model ('attention_unet', 'autoencoder', 'variational_autoencoder')
        **kwargs: Arguments passed to the model constructor
    """
    if model_name == 'attention_unet':
        return AttentionUNet(**kwargs)
    elif model_name == 'autoencoder':
        return ConvAutoencoder()
    elif model_name == 'variational_autoencoder':
        return VAE(**kwargs)
    elif model_name == 'bigger_variational_autoencoder':
        return Bigger_VAE(**kwargs)
    elif model_name == 'supervised_cnn':
        return SupervisedCNN()
    elif model_name == 'small_variational_autoencoder':
        return SmallVAE(**kwargs)
    elif model_name == 'micro_variational_autoencoder':
        return MicroVAE(**kwargs)
    elif model_name == 'nano_variational_autoencoder':
        return NanoVAE(**kwargs)
    elif model_name == 'even_smaller_variational_autoencoder':
        return EvenSmallerVAE(**kwargs)
    elif model_name == 'semi_supervised_vae':
        return SemiSupervisedVAE(**kwargs)
    else:
        raise ValueError(f"Unknown model name: {model_name}.")