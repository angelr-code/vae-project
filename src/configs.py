# This file contains the configuration for the VAE model based on the task it is being used for.

mnist_configs = {
        'input_dim': 784,
        'hidden_dims': [512, 256],
        'latent_dim': 2
}

brain_configs = {
        'input_dim': 65536,
        'hidden_dims': [1024,512,256],
        'latent_dim': 64
}

celeba_configs = {
    'input_dim': 49152,
    'hidden_dims': [2048, 1024, 512],
    'latent_dim': 256
}