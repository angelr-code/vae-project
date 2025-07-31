import torch
from torch import optim
from torch.utils.data import DataLoader 
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from pathlib import Path
from src.model import VAE
from src.configs import mnist_configs
from src.train import train
from src.utils import latent_visualization, print_manifold
from skimage.util import random_noise



'''Defining the dataset and training the model'''

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1)) # Flat input
])

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"

mnist = datasets.MNIST(root= DATA_DIR,
                       download = False,
                       train = True,
                       transform = transform)

train_loader = DataLoader(mnist,
                          batch_size=128,
                          shuffle=True,
                          pin_memory=True, # This will improve loading and training
                          drop_last=True # Last batch will be dropped if its size is not 64
                          )

model = VAE(**mnist_configs, f_out = 'sigmoid')

optimizer = optim.Adam(model.parameters(), lr= 3e-4)

train(model,train_loader,optimizer,device, max_beta = 4, epochs=50, f_out='sigmoid', labels=True)

'''Latent Space visualization'''

mnist_test = datasets.MNIST(root = DATA_DIR,
                            download=False,
                            train=False,
                            transform=transform)

test_loader = DataLoader(mnist_test,
                         batch_size=64,
                         shuffle=False,
                         pin_memory=True,
                         drop_last=False)

model.eval()
z = []
labels = []

with torch.no_grad():
    for x, y in test_loader:
        x = x.to(device)
        mu, _ = model.encoder(x)
        z.append(mu.cpu()) 
        labels.append(y)

z = torch.cat(z)
labels = torch.cat(labels)

latent_visualization(z, labels)

'''Generative Modeling'''

plt.rcdefaults() # Resets the Matplotlib settings to avoid unexpected results in the next plots

print_manifold(model, device)


'''Denoising and Imputation'''

def add_gaussian_noise(tensor, eps = 0.1):
    return tensor + torch.randn_like(tensor)*eps


def salt_and_pepper(x, amount):
    x = x.cpu().numpy()
    x = random_noise(x, mode='s&p', amount = amount)
    x = torch.tensor(x, dtype=torch.float32)
    return x


#Denoising
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1)),
    transforms.Lambda(lambda x:add_gaussian_noise(x))
])

mnist_noise = datasets.MNIST(root = DATA_DIR,
                            download=False,
                            train=False,
                            transform=transform)

noise_loader = DataLoader(mnist_noise,
                         batch_size=64,
                         shuffle=False,
                         pin_memory=True,
                         drop_last=False)


iterator = iter(noise_loader)

data, _ = next(iterator)

x = data[0].to(device)

corrupted_img = x.cpu().numpy()
corrupted_img = corrupted_img.reshape(28,28)


x_hat, _, _ = model(x)


with torch.no_grad():
    reconstruction_img = x_hat.cpu().numpy().reshape(28, 28)

plt.subplots(nrows = 1, ncols = 2)

plt.subplot(1, 2, 1)
plt.imshow(corrupted_img, cmap='gray')
plt.axis('off')


plt.subplot(1, 2, 2)
plt.imshow(reconstruction_img, cmap='gray')
plt.axis('off')
plt.tight_layout()
plt.show()


# Imputation

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1)),
    transforms.Lambda(lambda x:salt_and_pepper(x, amount = 0.15))
])

mnist_noise = datasets.MNIST(root = DATA_DIR,
                            download=False,
                            train=False,
                            transform=transform)

noise_loader = DataLoader(mnist_noise,
                         batch_size=64,
                         shuffle=False,
                         pin_memory=True,
                         drop_last=False)

iterator = iter(noise_loader)

data, _ = next(iterator)

x = data[0].to(device)

corrupted_img = x.cpu().numpy()
corrupted_img = corrupted_img.reshape(28,28)


x_hat, _, _ = model(x)

with torch.no_grad():
    reconstruction_img = x_hat.cpu().numpy().reshape(28, 28)


plt.subplots(nrows = 1, ncols = 2)

plt.subplot(1, 2, 1)
plt.imshow(corrupted_img, cmap='gray')
plt.axis('off')


plt.subplot(1, 2, 2)
plt.imshow(reconstruction_img, cmap='gray')
plt.axis('off')
plt.tight_layout()
plt.show()