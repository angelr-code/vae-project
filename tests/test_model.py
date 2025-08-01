import torch 
from src.model import Decoder, Encoder, VAE

def test_encoder_output_shape():
    """
    Test the output shape of the Encoder.

    The encoder should output two tensors: mu and logvar, both with shape (batch_size, latent_dim).
    """
    input_dim = 64
    hidden_dims = [32, 16]
    latent_dim = 8
    encoder = Encoder(input_dim, hidden_dims, latent_dim)

    batch_size = 2
    x = torch.randn(batch_size, input_dim)
    mu, logvar = encoder(x)

    assert mu.shape == (batch_size, latent_dim), "The dimensions of the means do not match"
    assert logvar.shape == (batch_size, latent_dim), "The dimensions of the logvars do not match"



def test_decoder_output_shape():
    """
    Test the output shape of the Decoder.

    The decoder should output a tensor with shape (batch_size, input_dim).
    """   
    input_dim = 64
    hidden_dims = [32, 16]
    latent_dim = 8
    decoder = Decoder(latent_dim, hidden_dims, input_dim, f_out = 'sigmoid')

    batch_size = 2
    z = torch.randn(batch_size, latent_dim)
    x_hat = decoder(z)

    assert x_hat.shape == (batch_size, input_dim), "The dimensions of the outputs do not match"


def test_vae_output_shape():
    """
    Test the output shape of the Variational Autoencoder (VAE).

    The VAE should output three tensors: x_hat, mu, and logvar.
    x_hat should have shape (batch_size, input_dim), while mu and logvar should have shape (batch_size, latent_dim).
    """
    input_dim = 64
    hidden_dims = [32, 16]
    latent_dim = 8
    model = VAE(input_dim, hidden_dims, latent_dim, f_out='sigmoid')

    batch_size = 2
    x = torch.randn(batch_size, input_dim)
    x_hat, _, _  = model(x)

    assert x_hat.shape == (batch_size, input_dim), "The dimensions of the outputs do not match"

def test_reparametrize():
    """
    Test the reparameterization trick in the VAE.

    The reparameterization trick should return a tensor with shape (batch_size, latent_dim).
    It should sample from the latent space using the mean and log-variance from the encoder.
    """
    input_dim = 64
    hidden_dims = [32, 16]
    latent_dim = 8
    model = VAE(input_dim, hidden_dims, latent_dim, f_out='sigmoid')

    batch_size = 2
    x = torch.randn(batch_size, input_dim)
    mu, logvar = model.encoder(x)

    z = model.reparameterize(mu, logvar)

    assert z is not None, "The reparametrization trick should return a tensor"
    assert isinstance(z, torch.Tensor), "The reparametrization trick should return a torch tensor"
    assert z.shape == (batch_size, latent_dim), "The reaparametrization trick outputs wrong dimensions"


