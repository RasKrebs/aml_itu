# -*- coding: utf-8 -*-
"""
Created on Mon Aug 11 12:35:22 2014

@author: gsubramanian
"""
import numpy as np
from scipy.optimize import minimize

epsilon = 1e-8


def soft_absolute(v):
    return np.sqrt(v**2 + epsilon)


def get_objective_fn(X,n_dim,n_features):
    def _objective_fn(W):
        W = W.reshape(n_dim,n_features)
        Y = np.dot(X,W)
        Y = soft_absolute(Y)
        
        # Normalize feature across all examples
        # Divide each feature by its l2-norm
        Y = Y / np.sqrt(np.sum(Y**2,axis=0) + epsilon)        
        
        # Normalize feature per example
        Y = Y / np.sqrt(np.sum(Y**2,axis=1)[:,np.newaxis] + epsilon )
        
        return np.sum(Y)
    return _objective_fn


def sfiltering(X,n_features=25):
    n_samples, n_dim = X.shape
    # Intialize the weight matrix W (n_dim,n_features)
    # Intialize the bias term b(n_features)
    W = np.random.randn(n_dim,n_features)
    obj_function = get_objective_fn(X,n_dim,n_features)
    
    opt_out = minimize(obj_function,W,method='L-BFGS-B',options={'maxiter':10,'disp':True})
    W_final = opt_out['x'].reshape(n_dim,n_features)
    
    transformed_x = np.dot(X,W_final)
    return transformed_x

def sfiltering_batch(X, n_features=5, batch_size=100):
    n_samples,n_dim = X.shape
    transformed_x = np.empty((n_samples, n_features))

    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        batch = X[start_idx:end_idx]
        transformed_batch = sfiltering(batch, n_features)
        transformed_x[start_idx:end_idx] = transformed_batch

    return transformed_x


import torch
from torch import nn
import torchvision
torchvision.disable_beta_transforms_warning()
import torchvision.transforms as T
from torchvision.transforms import v2
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader


# Setting up device
if torch.cuda.is_available():
    device = torch.device("cuda")
    print (f"GPU is available")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print('MPS device found.')
else:
    print ("No GPU available, using CPU instead")


# Sparse Filter Class (as previously defined)
# Sparse Filter Class
class SparseFilter(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SparseFilter, self).__init__()
        self.weights = nn.Parameter(torch.randn(input_dim, output_dim) * 0.5)
        self.epsilon = 1e-8

    def soft_abs(self, value):
        return torch.sqrt(value ** 2 + self.epsilon)

    def forward(self, x):
        first = torch.matmul(x, self.weights)
        second = self.soft_abs(first)
        third = second / torch.sqrt(torch.sum(second ** 2, axis=0) + self.epsilon)
        fourth = third / torch.sqrt(torch.sum(third ** 2, axis=1)[:, None] + self.epsilon)
        return torch.sum(fourth)

# Function to Load Weights and Create Model
def load_sparse_filter_model(input_dim, output_dim, model_path):
    model = SparseFilter(input_dim, output_dim)
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set to evaluation mode
    return model

# Assuming the path to the saved model is 'sparse_filter_model_weights.pth'
#input_dim = ...  # same as used during training
#output_dim = ... # same as used during training
#model_path = 'sparse_filter_model_weights_n25.pth'
#sparse_filter_model = load_sparse_filter_model(input_dim, output_dim, model_path)








