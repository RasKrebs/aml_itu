# Library imports
import torch
from math import ceil
from torch import nn
import torchvision
torchvision.disable_beta_transforms_warning()


    
def ResNet(model_load_path):
    # Load the model from pytorch
    resnet = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
    
    # Modify the last layer
    resnet.fc = nn.Sequential(
            nn.BatchNorm1d(resnet.fc.in_features, 512),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=512, out_features=10))

    # Load the model weights
    resnet.load_state_dict(torch.load(model_load_path, map_location=torch.device('cpu')))
    
    # Return the model
    return resnet
