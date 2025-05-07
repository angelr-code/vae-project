# This file contains the configuration for the VAE model based on the task it is being used for.

vae_configs = {
    'latent-manipulation' : {
        'input_dim': 784,
        'hidden_dims': [512, 256],
        'latent_dim': 64,
    }
}