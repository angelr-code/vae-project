import torch
from torch import optim
from PIL import Image
from pathlib import Path 
from src.model import VAE
from src.configs import brain_configs
from src.train import train
from src.utils import load_brain, visualize_brain_examples, visualize_heatmap

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ROOT_DIR = Path(__file__).resolve().parents[1]

"""Loading and visualizing training data"""
DATA_DIR = ROOT_DIR / "data" / "mri_brain_tumor" / "Training" / "notumor"
brain_loader = load_brain(DATA_DIR)
visualize_brain_examples(dataloader=brain_loader, num_examples=4, fig_size=(8,8))

"""Defining and training the model"""
model = VAE(**brain_configs, f_out = 'tanh')
optimizer = optim.Adam(model.parameters(), lr= 3e-4)
train(model,brain_loader,optimizer,device,epochs=50, f_out='tanh', max_beta = 1.5)

"""Testing"""
model.eval()

# meningioma
MENINGIOMA_IMG = ROOT_DIR / "data" / "mri_brain_tumor" / "Testing" / "meningioma" / "Te-meTr_0003.jpg"
image = Image.open(MENINGIOMA_IMG).convert('L')
visualize_heatmap(image, model, device, threshold=1, cmap='viridis')

# glioma 
GLIOMA_IMG = ROOT_DIR / "data" / "mri_brain_tumor" / "Testing" / "glioma" / "Te-gl_0026.jpg"
image = Image.open(GLIOMA_IMG).convert('L')
visualize_heatmap(image, model, device, threshold=0.58, cmap='viridis')

# pituitary
PITUITARY_IMG = ROOT_DIR / "data" / "mri_brain_tumor" / "Testing" / "pituitary" / "Te-pi_0036.jpg"
image = Image.open(PITUITARY_IMG).convert('L')
visualize_heatmap(image, model, device, threshold=1, cmap='viridis')

# no tumor
NOTUMOR_IMG = ROOT_DIR / "data" / "mri_brain_tumor" / "Testing" / "notumor" / "Te-no_0043.jpg"
image = Image.open(NOTUMOR_IMG).convert('L')
visualize_heatmap(image, model, device, threshold=1, cmap='viridis')
