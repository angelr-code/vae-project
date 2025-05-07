import torch
from torch.nn import functional as F

def loss(x, x_hat, mu, logvar):
    KL = 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())     # KL[q(z|x) || p(z)]
    BCE = F.binary_cross_entropy(x_hat, x.view(-1, 784), reduction='sum') # Reconstruction Loss

    return KL + BCE

def train(model, dataloader, optimizer, device, epochs = 10):
    model.to(device)
    model.train()
    for epoch in range(epochs):
        train_loss = 0.0
        for batch_idx, (data, _) in enumerate(dataloader):
            data = data.to(device)
            optimizer.zero_grad()
            x_hat, mu, logvar = model(data)
            loss_value = loss(data, x_hat, mu, logvar)
            loss_value.backward()
            optimizer.step()
            train_loss += loss_value.item()
        
        print(f'Epoch {epoch + 1}, Loss: {train_loss / len(dataloader.dataset)}')
        