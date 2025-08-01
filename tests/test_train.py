import torch
import pytest
from torch.utils.data import DataLoader, TensorDataset
from src.model import VAE
from src.train import loss, train

@pytest.fixture
def training_setup():
    """
    Fixture to set up the training environment for the VAE.
    It initializes a VAE model, a DataLoader with dummy data, and an optimizer.
    The input dimensions, hidden layers, and latent space dimensions are defined.

    Returns
    -------
    model : VAE
        The Variational Autoencoder model.

    dataloader : DataLoader
        DataLoader containing dummy data for training.

    optimizer : torch.optim.Optimizer
        Optimizer for the VAE model.

    input_dim : int
        Input dimension of the VAE.

    hidden_dims : list
        List of hidden layer dimensions.

    latent_dim : int
        Latent space dimension of the VAE.
    """
    input_dim = 64
    hidden_dims = [32, 16]
    latent_dim = 8
    batch_size = 4
    input_dim = 784
    hidden_dims = [32, 16]
    latent_dim = 8
    batch_size = 4

    data = (torch.rand(32, 1, 28, 28)).view(32, -1)   # MNIST-like dummy dataset
    labels = torch.randint(0, 10, (32,))    # Dummy labels for the dataset
    dataset = TensorDataset(data, labels)   
    dataloader = DataLoader(dataset, batch_size = batch_size)   

    model = VAE(input_dim, hidden_dims, latent_dim, f_out='sigmoid')    
    optimizer = torch.optim.Adam(model.parameters(), lr = 3e-4)

    return model, dataloader, optimizer, input_dim, hidden_dims, latent_dim

def test_train(training_setup):
    """
    Test the training function of the VAE.
    It checks if the training process runs without errors and if the model's weights change after training.

    Parameters
    ----------
    training_setup : fixture
        The setup containing the model, dataloader, optimizer, and other parameters.
    """

    # Unpack the training setup
    model, dataloader, optimizer, _, _, _ = training_setup
    initial_weights = model.encoder.hidden2mu.weight.clone()

    # Run the training function
    try:
        train(model, 
              dataloader, 
              optimizer, 
              device = 'cpu', 
              f_out = 'sigmoid', 
              max_beta = 4, 
              epochs = 1, 
              print_loss = False, 
              labels = True
        )
    except Exception as e:
        pytest.fail(f"The training function raised an exception: {e}")

    
    final_weights = model.encoder.hidden2mu.weight.clone()
    assert not torch.equal(initial_weights, final_weights)  # Check if weights have changed after training <-> Gradients are being applied

    