# This file contains the configuration for the VAE model based on the task it is being used for.

mnist_configs = {
        'input_dim': 784,
        'hidden_dims': [256,128,64,32],
        'latent_dim': 2
}