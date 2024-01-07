import numpy as np
import os
import torch
import time
import torch
from torch import nn
import torchvision
torchvision.disable_beta_transforms_warning()
import torchvision.transforms as T
from torchvision.transforms import v2
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import kornia
import cv2 as cv

os.environ["COLAB"] = "False"
# Changing directory into aml_itu
if os.getcwd().split('/')[-1] != 'aml_itu': os.chdir(os.path.abspath('.').split('aml_itu/')[0]+'aml_itu')

from utils.helpers import *
from utils.StatefarmPytorchDataset import StateFarmDataset

# Setting up device
if torch.cuda.is_available():
    device = torch.device("cuda")
    print (f"GPU is available")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print('MPS device found.')
else:
    print ("No GPU available, using CPU instead")
    
from utils.models.EfficientNet import EfficientNet
from utils.models.TinyVGG import TinyVGG
from utils.models.ResNet import ResNet
from utils.models.VGG16 import VGG16
from utils.models.EfficientNetb1 import EfficientNet_b1
from utils.pipelines.image_transformation import image_transformer
import argparse

# Get arguments
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default='tinyvgg', help="Model to use for prediction")
parser.add_argument("--size", type=str, default='L_SQUARED', help="Image size to use for prediction")
parser.add_argument("--seconds", type=int, default=10, help="Number of seconds to run prediction")
parser.add_argument("--weighted_frames", type=int, default=10, help="Number of seconds to run prediction")
parser.add_argument("--source", type=str, default='webcam', help="Source of video")
parser.add_argument("--device", type=str, default='gpu', help="Device to run on")
args = parser.parse_args()

# Load config
config = load_config(eval(os.environ["COLAB"]))

duration = 20  # duration of video in seconds

if args.device == 'cpu':
    device = torch.device('cpu')
else:
    # Setting up device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print (f"GPU is available")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print('MPS device found.')
    else:
        print ("No GPU available, using CPU instead")
        device = torch.device('cpu')
# Extract frame from camera
def get_frame():
    """Capture frame from webcam"""
    # Generate cam
    cam = cv.VideoCapture(0)
    _, frame = cam.read() 
    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    
    # Transform to tensor 
    frame = kornia.image_to_tensor(frame)
    return frame

# Load image from path
def load_image(path):
    """Load image from path"""
    frame = cv.imread(path)
    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    frame = kornia.image_to_tensor(frame)
    return frame

def batch_image(frame):
    """Transform image to batch"""
    frame = frame.unsqueeze(0)
    return frame

# Load model
if args.model == 'tinyvgg':
    # Path and directory files
    path = os.path.join(config['outputs']['path'], 'TinyVGG_500k')
    file = 'TinyVGG_500k_final.pt'
    
    # Load model
    model = TinyVGG()


elif args.model == 'efficientnetb0':
    # Path and directory files
    path = os.path.join(config['outputs']['path'], 'EfficientNet_after_CV')
    file = 'EfficientNet_final.pt'
    
    # Load model
    model = EfficientNet()

elif args.model == 'efficientnetb1':
    # Path and directory files
    path = os.path.join(config['outputs']['path'], 'EfficientNet')
    file = 'efficientnet_b1_94.pt'
    
    # Load model
    model = EfficientNet_b1()
    
    
elif args.model == 'vgg16':
    # Path and directory files
    path = os.path.join(config['outputs']['path'], 'VGG16')
    file = 'VGG16_v3_20231229_183229_epoch_10.pt'
    
    # Load model
    model = VGG16()

elif args.model == 'resnet':
    # Path and directory files
    path = os.path.join(config['outputs']['path'], 'ResNet')
    file = 'ResNet_v2_20240103_013959_epoch_14_highest_acc.pt'
    
    # Load model
    model = ResNet()
    
else:
    raise ValueError('Model not found')

# Load model
model.load_state_dict(torch.load(os.path.join(path, file), map_location=torch.device('cpu')))
model.to(device)
model.eval()

# Get image size
size = args.size.upper()

# Predefined image sizes
def inference_loop(model, 
                   image_size,
                   weighted_frames = 10,
                   total_seconds=10,
                   device=device):
    """Inference loop for a given model"""
    
    # Inference loop helpers
    model = model.to(device)
    predictions = weighted_prediction(config, n=weighted_frames)
    start_time = time.time()
    font = cv.FONT_HERSHEY_SIMPLEX
    font_scale = .75  # Font size
    thickness = 2  # Font thickness
    top_border_size = 50
    border_color = (255, 255, 255)  # White color in BGR format
    
    # Add external window if webcam is used
    if args.source == 'webcam':
        # Create a named window
        cv.namedWindow('Output', cv.WINDOW_NORMAL)

        # Resize the window
        window_width = 1200
        window_height = 1200
        cv.resizeWindow('Output', window_width, window_height)
    
    while True:
        if args.source == 'webcam':
            # Get frame
            frame = get_frame()
        else:
            # Get frame
            frame = load_image(args.source)
        
        # Transform image
        frame = image_transformer(frame, size=image_size)
        #frame = T.RandomHorizontalFlip(p=1)(frame)
       
        # Transform to batch
        frame = batch_image(frame)
       
       
        # Append prediction and compute prediction time
        prediction_start_time = time.time()
        prediction = model(frame.to(device)) 
        prediction_end_time = time.time()
        prediction_time = prediction_end_time - prediction_start_time
        
        if args.source != 'webcam':
            out = np.argmax(prediction.detach().cpu().numpy())
            # Extract prediction
            text = predictions.map_labels(out)

            # Print prediction time and prediction
            print(f'Prediction time: {prediction_time} ({round(1/prediction_time)} fps). Prediction: {text}')
            break
        
        else:
            pass
        out = predictions(prediction)
        # Image to numpy
        frame = cv.cvtColor(frame.squeeze(0).detach().cpu().permute(1, 2, 0).numpy(), cv.COLOR_RGB2BGR)
        
        # Add the border on top
        frame = cv.copyMakeBorder(frame, top=top_border_size, bottom=0, left=0, right=0, 
                                  borderType=cv.BORDER_CONSTANT, value=border_color)
        # Extract prediction
        if out is not None:
            text = predictions.map_labels(out)
        else: 
            text = 'Out of scope'
        
        # Print prediction time and prediction
        print(f'Prediction time: {prediction_time} ({round(1/prediction_time)} fps). Prediction: {text}')

        # Position the text
        textX = 300 
        textY = 15  # Position the text 30 pixels from the top edge
        
        # Put the text on the image
        cv.putText(frame, text, (textX, textY), font, font_scale, (0,0,255), thickness)
        
        # Show image
        cv.imshow('Output', frame)
        
        
        # If total seconds have passed, break
        if time.time() - start_time > total_seconds:
            break
        
        # Wait for 25 ms and check if the user wants to exit (press 'q')
        if cv.waitKey(25) & 0xFF == ord('q'):
            break
    
    return predictions
    
inference_predictions = inference_loop(model, image_size=args.size, weighted_frames=int(args.weighted_frames), total_seconds=int(args.seconds))