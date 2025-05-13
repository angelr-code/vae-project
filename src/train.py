import torch
from torch.nn import functional as F



def loss(x, x_hat, mu, logvar, beta, input_dim, f_out):
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

        beta: float
            Used to control the influenced of KL divergence in the loss function expression.

        input_dim: int
            Dimension of the flattened image tensors.

        f_out: str
            Indicates the output layer activation.
            Must be 'sigmoid' or 'tanh'.

    Returns
    -------
        -ELBO: torch.Tensor
    """
    
    
    KL = - 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())     # KL[q(z|x) || p(z)]


    # Depending on the output layer activation we choose the suitable reconstruction error
    if f_out == 'sigmoid':
        reconstruction_error = F.binary_cross_entropy(x_hat, x.view(-1, input_dim), reduction='sum') # Reconstruction Loss. Binary Cross Entropy
    else:
        reconstruction_error = F.mse_loss(x_hat, x.view(-1, input_dim), reduction='sum') # Reconstruction Loss. Mean Squared Error


    return beta*KL + reconstruction_error, KL, reconstruction_error



def train(model, dataloader, optimizer, device, f_out, epochs = 20, print_loss = True, labels = False):
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

        f_out: str
            Indicates the output layer activation.
            Must be 'sigmoid' or 'tanh'.

        epochs: int 
            Number of training epochs. Default: 20.

        print_loss: bool 
            If True, shows the training loss per epoch. Default: True

        labels: bool
            If true, the data contains labels that must be ignored. Default: False
    """

    #If activation value is unexpected we raise an error.
    if f_out != 'sigmoid' and f_out != 'tanh':
        raise ValueError("The output activation must be either 'sigmoid' or 'tanh'")

    model.to(device)
    model.train() # Training mode
    
    beta = 1/epochs
    max_beta = 4.0

    for epoch in range(epochs):
        train_loss = 0.0
        
        # We iterate through the mini-batches
        for _, data in enumerate(dataloader):  
            
            if labels:
                x, _ = data #In some datasets like MNIST there are labels, we ignore them in the VAE.
            else:
                x = data

            x = x.to(device)
            optimizer.zero_grad()   # Sets all loss function gradients to zero
            x_hat, mu, logvar = model(x)
            loss_value, kl, reconstruction_error = loss(x, x_hat, mu, logvar, beta, input_dim = x.shape[1], f_out = f_out)
            loss_value.backward() # Loss Backprop. New gradients
            optimizer.step()    # Updates the model parameters following the chosen optimizer scheme
            train_loss += loss_value.item() # The loss is in torch.Tensor format, we transform it to float

        beta = min((epoch+1)/epochs, 1.0) * max_beta # Beta parameter annealing
        
        if print_loss:
            print(f'Epoch {epoch + 1} -----> Loss: {train_loss / len(dataloader.dataset)} | KL: {kl} | Reconstruction Error: {reconstruction_error}')
        