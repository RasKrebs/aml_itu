# Library import
import os
import numpy as np
import yaml

"""Methods in this file are used to help with the project. Any function applied frequently is best stored here"""

# Function for loading the config file
def load_config(colab = False) -> dict:
    """Function for loading the config file

    Args:
        colab (bool, optional): Use if project is project is being run on colab. Defaults to False.

    Returns:
        dict: Dictionary with project configuration settings
    """
    
    # Ensuring that working directory is appropriately set
    assert os.getcwd().split('/')[-1] == 'aml_itu', 'Working directory not set to root of project. Currently working in: ' + os.getcwd()    
    
    # Loading the config file
    with open('config.yml') as file:
        config = yaml.safe_load(file)
    
    # Setting the dataset path depending on environment
    if colab: 
        # Setting the colab path
        colab_path = config['dataset']['colab_path']
        
        # Replace the '..' in the path with the colab path
        config['dataset']['data'] = config['dataset']['data'].replace('..', colab_path)
        config['dataset']['images']['train'] = config['dataset']['images']['train'].replace('..', colab_path)
        config['dataset']['images']['test'] = config['dataset']['images']['test'].replace('..', colab_path)

    return config


class weighted_prediction:
    def __init__(self, config, n=10):
        """Class takes pytorch predictions as input and outputs a weighted prediction over the last n frames"""
        self.n = n
        self.config = config
        self.predictions = None
        self.weighted_predictions = []

    # Append prediction to list
    def __call__(self, prediction):
        """Performs the weighted average"""
        # Append prediction to list
        if self.predictions is None:
            self.predictions = prediction.detach().cpu().numpy()
        else:
            self.predictions = np.vstack((self.predictions, prediction.detach().cpu().numpy()))

        # If their arent enought predictions, return nothing
        if self.predictions.shape[0] < self.n:
            self.weighted_predictions.append(None)
            return None
        
        # Else return the weighted prediction over the last n frames
        else:
            self.weighted_predictions.append(np.argmax(self.predictions[-self.n:, :].mean(axis=0)))
            return self.weighted_predictions[-1]
            
    def map_labels(self, prediction):
        """Maps the prediction to the correct class"""
        if prediction is None:
            return 'Out of scope'
        return self.config['dataset']['class_mapping'][f'c{prediction}']