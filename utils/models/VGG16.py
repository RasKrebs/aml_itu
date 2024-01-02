# Library imports
import torch
import torch
from math import ceil
from torch import nn
import torchvision
torchvision.disable_beta_transforms_warning()

vgg_16 = torchvision.models.vgg16(pretrained=False)
vgg_16.classifier[6] = nn.Sequential(
                        nn.Linear(4096, 256),
                        nn.ReLU(),
                        nn.Dropout(0.6),
                        nn.Linear(256, 10)
                        )
    
model_load_path = '' #tbd by you rasmus

vgg_16.load_state_dict(torch.load(model_load_path, map_location=torch.device('cpu')))