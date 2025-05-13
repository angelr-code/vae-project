import os
from PIL import Image
import numpy as np 
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
import torch
import ipywidgets as widgets
from matplotlib.animation import FuncAnimation
from IPython.display import display, HTML



# CelebA dataset PyTorch load utils.

class CeleADataset(Dataset):
    """
    Custom celebA dataset for unsupervised learning tasks. It inherits from the torch.utils.data Dataset Module.

    Parameters
    ----------

    root_dir: str
        Path to the directory containing the dataset images.
    
    transforms: callable, optional  
        Transformation methods applied to raw image data (usually a composition
        of torchvision.transforms). Default is None.
    """

    # A torch custom Dataset class must always implement the three functions: __init__, __len__, and __getitem__.

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith(".jpg")]  # Avoids getting corrupted files.

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        """Gets an Image from the dataset at a given index"""

        img_name = os.path.join(self.root_dir, self.image_files[idx])
        
        
        # We open and convert the image to RGB using PIL
        try:
            image = Image.open(img_name).convert("RGB")
        except Exception as e:
            print(f"Error loading image {img_name}: {e}")
            return None
        
        # Applies the image the transformation methods if given.
        if self.transform:
            image = self.transform(image)

        image = image.view(-1)  # Flattens the Image

        return image
    


def load_celeba(root_dir, batch_size = 64, image_size = 64, num_workers = 4):
    """
    Creates the CelebA torch DataLoader.

    Parameters
    ----------
    root_dir: str
        Path to the directory containing the dataset images.

    batch_size: int
        Size of the training mini-batches. Default is 64.

    image_size: int 
        Image size resolution in pixels (img_size x img_size). Default is 64.
    
    num_workers: int 
        Number of paralel subprocesses used for data loading.
        Higher values may speed up data loading but use more CPU resources. Default is 4. 

    Returns
    -------
    dataloader: torch.utils.data.DataLoader
        CelebA torch dataloader. 
    """

    # Dataset transformations 
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)) # We normalize the inputs to [-1, 1]. The 3 dimensions are because of the 3 RGB channels.
    ])

    dataset = CeleADataset(root_dir = root_dir, transform = transform)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers= num_workers,
        pin_memory=True, #True because I will use CUDA. 
        drop_last=True # If len(last mini-batch) != batch_size it will be dropped.
    )

    return dataloader


def denormalize(tensor):
    """
    Denormalizes the loaded CelebA images for representation purposes.

    Parameters
    ----------
    tensor: torch.Tensor
        Tensor to be denormalized.

    Returns
    -------
    torch.Tensor
        Denormalized tensor.
    """

    if len(tensor.shape) == 4:
        result = tensor.clone().detach()
        result = (result * 0.5 ) + 0.5 # Inverse transformation
        return result
    else:
        # Single image
        return (tensor * 0.5) + 0.5
    

def visualize_celeba_examples(dataloader, num_examples, img_size = 64,  fig_size = (15,15)):
    """
    Function to visualize a small subset of CelebA dataset samples.

    Parameters
    ----------
    dataloader: torch.utils.data.DataLoader
        CelebA dataloader process.
    
    num_examples: int
        Number of samples to be shown. Must be smaller than the batch size.

    image_size: int 
        Image size resolution in pixels (img_size x img_size). Default is 64.

    fig_size: tuple
        Plot size. Default is (15,15).    
    """

    # Gets a minibatch and num_examples elements from it
    dataiter = iter(dataloader)
    images = next(dataiter)

    try:
        images = images[:num_examples]
    except Exception as e:
        print(f"The argument num_examples cannot be grater than the batch size. Error: {e}")

    images = images.view(num_examples, 3, img_size, img_size) # Unflatens the image and gets a matrix for each RGB channel.

    images = denormalize(images) 

    nrow = int(np.sqrt(num_examples)) # number of rows in the plot grid

    grid = torchvision.utils.make_grid(images, nrow=nrow, padding=2)

    # We move the grid tensor to the CPU and transform it into a NumPy array 
    grid_np = grid.cpu().numpy().transpose((1,2,0))

    plt.figure(figsize=fig_size)
    plt.imshow(grid_np)
    plt.axis('off')
    plt.title('CelebA Examples')
    plt.show()