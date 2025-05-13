import torch 
from torch import nn

class Encoder(nn.Module):
    """
    Encoder network for the Variational Autoencoder (VAE).

    Maps the input data into the parameters of a Gaussian distribution
    in the latent space.

    It inherits from the torch nn Module.

    Parameters
    ----------
    input_dim: int
        input dimension of the VAE.

    hidden_dims: int list
        list containing the hidden layers dimensions.

    latent_dim: int 
        latent space dimension of the VAE.
    """

    def __init__(self, input_dim, hidden_dims, latent_dim):
        super().__init__()
        
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            in_dim = h_dim
        
        self.encoder = nn.Sequential(*layers)
        self.hidden2mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.hidden2logvar = nn.Linear(hidden_dims[-1], latent_dim) 

        # We model the log-variance to ensure the variance remains positive 
        # after applying the exponential function.

    def forward(self, x):
        """
        Forward pass through the encoder.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, input_dim).

        Returns
        -------
        mu : torch.Tensor
            Mean of the approximate posterior.

        logvar : torch.Tensor
            Log-variance of the approximate posterior.
        """

        h = self.encoder(x)
        mu = self.hidden2mu(h)
        logvar = self.hidden2logvar(h)
        return mu, logvar
        
class Decoder(nn.Module):
    """
    Decoder network for the Variational Autoencoder (VAE).

    Maps the latent space representation into the data reconstruction (output).

    It inherits from the torch nn Module.

    Parameters
    ----------
    latent_dim: int 
        latent space dimension.

    hidden_dims: int list
        list containing the hidden layers dimensions.

    output_dim: int
        output dimension of the VAE.

    f_out: str
        Indicates the output layer activation.
        Must be 'sigmoid' or 'tanh'.
    """

    def __init__(self, latent_dim, hidden_dims, output_dim, f_out):
        super().__init__()
        layers = []

        in_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            in_dim = h_dim
        
        self.decoder = nn.Sequential(*layers)
        self.hidden2out = nn.Linear(hidden_dims[0], output_dim)

        #If activation value is unexpected we raise an error.
        if f_out != 'sigmoid' and f_out != 'tanh':
            raise ValueError("The output activation must be either 'sigmoid' or 'tanh'")
        
        self.f_out = f_out


    def forward(self, z):
        """
        Forward pass through the decoder.

        Parameters
        ----------
        z : torch.Tensor
            Input tensor of shape (batch_size, latent_dim).

        Returns
        -------
        x_hat : torch.Tensor
            Output tensor. 
        """

        h = self.decoder(z)

        if self.f_out == 'sigmoid':
            x_hat = torch.sigmoid(self.hidden2out(h))
        else:
            x_hat = torch.tanh(self.hidden2out(h))
            
        return x_hat
    

class VAE(nn.Module):    
    """
    Variational Autoencoder (VAE) Architecture.
    The approximate posterior is a factorized multidimensional Gaussian distribution.

    Maps the input data into its reconstruction. It combines the Encoder 
    and Decoder Modules.

    It inherits from the torch nn Module.

    Parameters
    ----------
    input_dim: int
        input dimension of the VAE

    hidden_dims: int list
        list containing the hidden layers dimensions

    latent_dim: int 
        latent space dimension  
            
    f_out: str
        Indicates the output layer activation.
        Must be 'sigmoid' or 'tanh'.  
    """

    def __init__(self, input_dim, hidden_dims, latent_dim, f_out):
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dims, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dims, input_dim, f_out) # In the VAE the input dimension matches the output dimension

    def reparameterize(self, mu, logvar):
        """
        Samples from the latent space using the reparametrization trick.

        Parameters
        ----------
        mu : torch.Tensor
            Mean of the approximate posterior.

        logvar : torch.Tensor
            Log-variance of the approximate posterior.  
        
        Returns
        -------
        torch.Tensor
            Latent space samples.

        """

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        """
        Forward pass through the Variational Autoencoder (VAE).

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, input_dim).

        Returns
        -------
        x_hat : torch.Tensor
            Data reconstruction (output). 

        mu : torch.Tensor
            Mean of the approximate posterior.

        logvar : torch.Tensor
            Log-variance of the approximate posterior.
        """

        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decoder(z)
        return x_hat, mu, logvar