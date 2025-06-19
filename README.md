## Project Overview

This repository contains the implementation of a Variational Autoencoder (VAE) using PyTorch. It was developed as part of my undergraduate final thesis (TFG). It implements a VAE in PyTorch that can be used for image generation, latent manipulation, anomaly detection and latent space exploration, etc. It includes:

- Encoder and decoder architectures (`src/model.py`)
- Configurable training via `src/train.py`
- Reusable helper functions (`utils/`)
- Applications to different problems (`notebooks/`)
- Model Configurations depending on application (`src/configs.py`)