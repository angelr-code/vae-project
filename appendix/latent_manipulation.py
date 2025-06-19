import sys 
import os 

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

import torch
import pandas as pd
from src.model import VAE
from src.utils import load_celeba, visualize_celeba_examples, denormalize, image_reconstruction, latent_interpolation
from src.configs import celeba_configs
from PIL import Image
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""Training examples visualization"""



root = r"C:\Users\Ángel\Documents\GitHub\vae-project\notebooks\data\celeba\img_align_celeba"
celeba_loader = load_celeba(root)
visualize_celeba_examples(dataloader=celeba_loader, num_examples=9, img_size=128,  fig_size=(12,12))


model = VAE(**celeba_configs, f_out='tanh').to(device)
model.load_state_dict(torch.load(r"C:\Users\Ángel\Documents\GitHub\vae-project\src\vae_celeba_trained2.pth", map_location= device))


"""Testing"""


model.eval()

img_root = r"C:\Users\Ángel\Documents\GitHub\vae-project\notebooks\data\celeba\test\000012.jpg"
image = Image.open(img_root).convert('RGB')
plt.imshow(image)
plt.axis('off')
plt.title('Imagen Original')
plt.show()

image_reconstruction(img_root, model, device)


"""Latent directions calculation"""



# This may take a while

df = pd.read_csv(r"C:\Users\Ángel\Documents\GitHub\vae-project\notebooks\data\celeba\list_attr_celeba.csv", sep = ',')
columns = ['Smiling', 'Male', 'Blond_Hair', 'No_Beard', 'Young', 'Eyeglasses']


df = df.set_index('image_id')
df = df[columns]
df.rename(columns={'No_Beard': 'Beard'}, inplace = True) 
df['Beard'] = df['Beard']*(-1)

root = r"C:\Users\Ángel\Documents\GitHub\vae-project\notebooks\data\celeba\img_align_celeba"

files_per_attribute = {}

for column in df.columns:
    files_per_attribute[column] = df[df[column] == 1].index.tolist()
    files_per_attribute[f'no_{column}'] = df[df[column] == -1].index.tolist()


loaders = {}

for attribute, files in files_per_attribute.items():
    loaders[attribute] = load_celeba(root, image_files = files)


def encode_latents(dataloader, model, device):
    latents = []

    with torch.no_grad():
        for imgs in dataloader:
            imgs = imgs.to(device)
            imgs = imgs.view(imgs.size(0), -1)
            mu, _ = model.encoder(imgs)
            latents.append(mu)
    
    return torch.cat(latents, dim = 0)

latents = {}

for attribute, loader in loaders.items():
    z = encode_latents(loader, model, device)
    mean = z.mean(dim = 0)
    latents[attribute] = mean


smile_dir = latents['Smiling'] - latents['no_Smiling']
smile_dir.to(device)

male_dir = latents['Male'] - latents['no_Male']
male_dir.to(device)

blond_dir = latents['Blond_Hair'] - latents['no_Blond_Hair']
blond_dir.to(device)

beard_dir = latents['Beard'] - latents['no_Beard']
beard_dir.to(device)

young_dir = latents['Young'] - latents['no_Young']
young_dir.to(device)

eyeglasses_dir = latents['Eyeglasses'] - latents['no_Eyeglasses']
eyeglasses_dir.to(device)

directions = {
    'smile': smile_dir,
    'male': male_dir,
    'blond': blond_dir,
    'beard': beard_dir,
    'young': young_dir,
    'eyeglasses': eyeglasses_dir
}



"""Visualizations"""



#Original
_, modified_img = latent_interpolation(model, device, directions, img_root, 0,0,0,0,0,0)
plt.imshow(modified_img)
plt.axis('off')
plt.show()


# No Smile
_, modified_img = latent_interpolation(model, device, directions, img_root, smile = -1.5, male = 0, blond = 0, beard = 0, young = 0, glasses = 0)
plt.imshow(modified_img)
plt.axis('off')
plt.show()

# old
_, modified_img = latent_interpolation(model, device, directions, img_root, smile = 0, male = 0, blond = 0, beard = 0, young = -3.5, glasses = 0)
plt.imshow(modified_img)
plt.axis('off')
plt.show()

# beard
_, modified_img = latent_interpolation(model, device, directions, img_root, smile = 0, male = 0, blond = .5, beard = 3, young = 0, glasses = 0)
plt.imshow(modified_img)
plt.axis('off')
plt.show()

# Smile and glasses
_, modified_img = latent_interpolation(model, device, directions, img_root, smile = 1, male = 0, blond = 0, beard = 0, young = 2, glasses = 3.5)
plt.imshow(modified_img)
plt.axis('off')
plt.show()

# blond woman with no smile
_, modified_img = latent_interpolation(model, device, directions, img_root, smile = -1, male = -1.5, blond = 0.75, beard = 0, young = 0, glasses = 0)
plt.imshow(modified_img)
plt.axis('off')
plt.show()