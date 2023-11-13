# Library Import
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset
from torchvision.io import read_image
import os

# Custom PyTorch Dataset Class
class StateFarmDataset(Dataset):
    def __init__(self, config, transform=None, target_transform=None, type:str = None):
        """
        Custom PyTorch Dataset Class.
        Args:
            config (dict): configuration dictionary. Loaded using load_config() function from utils/helpers.py
            transform (callable, optional): Optional transform to be applied on a sample.
            target_transform (callable, optional): Optional transform to be applied on a label.
            type (callable, optional): Optional specification of type of dataset. Can be either 'train', 'test' or None, which includes all.
        """
        
        # Ensuring directory corresponds to root of repo
        while not os.getcwd().split('/')[-1] == 'aml_itu':
            os.chdir('../')

        # Generating data variables
        self.config = config
        self.test_subjects = ['p051', 'p014', 'p015', 'p035', 'p047']
        self.metadata = pd.read_csv(self.config['dataset']['data'])
        
        # Extracting training or test data
        if type == 'train':
            self.metadata = self.metadata[-self.metadata['subject'].isin(self.test_subjects)]
        elif type == 'test':
            self.metadata = self.metadata[self.metadata['subject'].isin(self.test_subjects)]
        else:
            pass
        
        # Class mappings
        self.id_to_class = self.config['dataset']['class_mapping']
        
        # Appending target column with mapped class names
        self.metadata['target'] = self.metadata['classname'].map(self.id_to_class)
        
        # Path to img directory
        self.img_dir = self.config['dataset']['images']['train']
        self.metadata['img_path'] = self.img_dir + '/' + self.metadata['classname'] + '/' + self.metadata['img']
        self.img_labels = self.metadata[['img', 'classname', 'img_path']]
        
        # Data transformations
        self.transform = transform
        self.target_transform = target_transform

    # Returns length of dataset
    def __len__(self):
        return len(self.img_labels)

    # Returns image and label
    def __getitem__(self, idx):
        """
        Function for returning image and label.
        Args:
            idx (int): index of the sample to return.
        """
        
        # Extract image path
        img_path = self.img_labels.iloc[idx, 2]
        # Read image
        image = read_image(img_path)
        # Extract label
        label = self.img_labels.iloc[idx, 1]
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        
        return image, label
    
    def display_classes(self, 
                        seed:int = None, 
                        transform = None, 
                        id_to_class:bool = False,
                        figsize:tuple =(15, 10)):
        """Function for displaying randomg samples from each class"""
        
        # Set seed if specified
        if seed: np.random.seed(seed)
        
        # Extract random image per class
        self.imgs = self.img_labels.groupby('classname').sample(1).reset_index(drop=True)
        
        # Generate figure
        fig, axes = plt.subplots(3, 3, figsize=figsize)
        
        # Loop through axes and display images
        for i, ax in enumerate(axes.flat):
            img = Image.open(self.imgs.iloc[i, 2])
            ax.imshow(img)
            if id_to_class: ax.set_title(self.id_to_class[self.imgs.iloc[i, 1]])
            else: ax.set_title(self.imgs.iloc[i, 1])
            
            ax.axis('off')
        # Apply tight layout
        fig.tight_layout()
        
    def __repr__(self) -> str:
        return ' '.join(self.config['dataset']['name'].split('-')).title() + ' Dataset'