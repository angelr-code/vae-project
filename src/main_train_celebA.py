from model import VAE
from utils import load_celeba
from configs import celeba_configs
from train import train
import torch 
from torch import optim


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    celeba_loader = load_celeba(r"C:\Users\Ángel\Documents\GitHub\vae-project\notebooks\data\celeba\img_align_celeba", num_workers = 8)

    model = VAE(**celeba_configs, f_out='tanh').to(device)
    optimizer = optim.Adam(model.parameters(), lr=3e-4)

    train(model, celeba_loader, optimizer, device, f_out='tanh', epochs=100, max_beta = 2)

    torch.save(model.state_dict(), 'vae_celeba_trained.pth')

if __name__ == "__main__":
    main()