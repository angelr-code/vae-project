import sys 
import os 

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

import torch
from torch import optim
from PIL import Image
from src.model import VAE
from src.configs import brain_configs
from src.train import train
from src.utils import load_brain, visualize_brain_examples, visualize_heatmap

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

root = r"C:\Users\Ángel\Documents\GitHub\vae-project\notebooks\data\mri_brain_tumor\Training\notumor"
brain_loader = load_brain(root)

visualize_brain_examples(dataloader=brain_loader, num_examples=4, fig_size=(8,8))


model = VAE(**brain_configs, f_out = 'tanh')

optimizer = optim.Adam(model.parameters(), lr= 3e-4)

train(model,brain_loader,optimizer,device,epochs=50, f_out='tanh', max_beta = 1.5)

model.eval()

# meningioma
test_root = r"C:\Users\Ángel\Documents\GitHub\vae-project\notebooks\data\mri_brain_tumor\Testing\meningioma\Te-meTr_0003.jpg"
image = Image.open(test_root).convert('L')

visualize_heatmap(image, model, device, threshold=1, cmap='viridis')

# glioma 
test_root = r"C:\Users\Ángel\Documents\GitHub\vae-project\notebooks\data\mri_brain_tumor\Testing\glioma\Te-gl_0026.jpg"
image = Image.open(test_root).convert('L')

visualize_heatmap(image, model, device, threshold=0.58, cmap='viridis')


# Pituitary

test_root = r"C:\Users\Ángel\Documents\GitHub\vae-project\notebooks\data\mri_brain_tumor\Testing\pituitary\Te-pi_0036.jpg"
image = Image.open(test_root).convert('L')

visualize_heatmap(image, model, device, threshold=1, cmap='viridis')

# No Tumor

test_root = r"C:\Users\Ángel\Documents\GitHub\vae-project\notebooks\data\mri_brain_tumor\Testing\notumor\Te-no_0043.jpg"
image = Image.open(test_root).convert('L')

visualize_heatmap(image, model, device, threshold=1, cmap='viridis')