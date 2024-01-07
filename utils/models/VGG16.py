# Library imports
import torch
from math import ceil
from torch import nn
import torchvision
torchvision.disable_beta_transforms_warning()
    
def VGG16(model_load_path=None):
    # Load the model from pytorch
    vgg_16 = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.DEFAULT)
    
    # Modify the last layer
    vgg_16.classifier[6] = nn.Sequential(
                        nn.Linear(4096, 256),
                        nn.ReLU(),
                        nn.Dropout(0.6),
                        nn.Linear(256, 10))
    
    if model_load_path is None:
        return vgg_16
    else:
        
        # Load the model weights
        vgg_16.load_state_dict(torch.load(model_load_path, map_location=torch.device('cpu')))
        
        return vgg_16