import torch
from torch.nn import functional as F

def loss(x, x_hat, mu, logvar):
    """
    Computes the Variational Autoencoder (VAE) negative ELBO given datapoints and their reconstruction.
    Thus, minimizing this expresion will be equivalent to maximizing the ELBO.

    Computes both the KL divergence KL[q(z|x) || p(z)] and the reconstruction loss and then adds
    them up to get the ELBO.

    Parameters
    ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, input_dim).

        x_hat: torch.Tensor
            VAE x reconstructionm (output) (bath_size, input_dim).

        mu : torch.Tensor
            Mean of the approximate posterior (batch_size, latent dim).

        logvar : torch.Tensor
            Log-variance of the approximate posterior (batch_size, latent dim).  

    Returns
    -------
        - ELBO: torch.Tensor
    """
    
    KL = - 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())     # KL[q(z|x) || p(z)]
    MSE = F.mse_loss(x_hat, x.view(-1, 12288), reduction='sum') # Reconstruction Loss. Mean squared error

    # ELBO = - KL + Reconstruction Error

    return  KL - MSE

def train(model, dataloader, optimizer, device, epochs = 20, print_loss = True):
    """
    Trains the Variational Autoencoder (VAE) maximizing the ELBO.

    Parameters
    ----------

        model: nn.Module
            VAE model to be trained.

        dataloader: torch.utils.data.DataLoader
            DataLoader which loads the training data.

        optimizer: torch.optim
            Optimization method.

        device: torch.device
            Training device (GPU or CPU).

        epochs: int 
            Number of training epochs. Default: 20.

        print_loss: bool 
            If True, shows the training loss per epoch. Default: True
    """

    model.to(device)
    model.train() # Training mode

    for epoch in range(epochs):
        train_loss = 0.0
        
        # We iterate through the mini-batches
        for _, data in enumerate(dataloader):  
            data = data.to(device)
            optimizer.zero_grad()   # Sets all loss function gradients to zero
            x_hat, mu, logvar = model(data)
            loss_value = loss(data, x_hat, mu, logvar)
            loss_value.backward() # Loss Backprop. New gradients
            optimizer.step()    # Updates the model parameters following the chosen optimizer scheme
            train_loss += loss_value.item() # The loss is in torch.Tensor format, we transform it to float
        
        if print_loss:
            print(f'Epoch {epoch + 1}, Loss: {train_loss / len(dataloader.dataset)}')
        