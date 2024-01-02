# Library imports
import torch
import torch
from math import ceil
from torch import nn
import torchvision
torchvision.disable_beta_transforms_warning()

resnet = torchvision.models.resnet18(pretrained=False)
resnet.fc = nn.Sequential(
            nn.BatchNorm1d(resnet.fc.in_features, 512),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=512, out_features=10)
)

model_load_path = '' #tbd by you rasmus

resnet.load_state_dict(torch.load(model_load_path, map_location=torch.device('cpu')))