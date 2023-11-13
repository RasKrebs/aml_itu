# Library import
import os
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