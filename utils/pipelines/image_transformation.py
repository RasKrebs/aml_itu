import torch
from torch import nn
import torchvision.transforms as T
from torchvision.transforms import v2




def image_transformer(img, 
                      image_augmentations = 'default', 
                      size='L'):
    """Apply dataaugmentation and load image into dataloader for inference

        args: 
            img: Image to be transformed
            image_augmentation (optional): Alternative image augmentations to apply. Defaults to resizing and to torch transformations.
            target_transformations (optional): Alternative target transformation. Defaults to removing c.
    """
    # Ensure size is capital
    size = size.upper()
    
    # Predefined image sizes
    img_sizes = {
        'S': (48, 64),
        'M': (93, 124),
        'L':(168, 224),
        'L_SQUARED':(224, 224),
    }
    
    # Checking that size exists on list
    assert size in ['L_SQUARED', 'L','M','S'], f"Image size must be one of: S ({img_sizes['S']}), M ({img_sizes['M']}), L ({img_sizes['L']}) or L_SQUARED ({img_sizes['L_SQUARED']})"
    
    # Image augmentations - apply default if none is specified
    if image_augmentations == 'default':
        image_augmentations = v2.Compose([T.Resize(img_sizes[size], antialias=True),
                                    v2.ToDtype(torch.float32, scale=True)])
    
    return image_augmentations(img)
    