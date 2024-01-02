# Library imports
import timm
from torch import nn
import torch

    
def EfficientNet_b1(model_load_path):
    # Load the model from timm
    efficientb1 = timm.create_model('efficientnet_b1', pretrained=False)
    
    # Modify the last layer
    num_ftrs = efficientb1.classifier.in_features
    efficientb1.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, 10)
            )

    # Load the model weights
    efficientb1.load_state_dict(torch.load(model_load_path, map_location=torch.device('cpu')))
    
    return efficientb1