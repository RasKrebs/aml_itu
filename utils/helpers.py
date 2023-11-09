# Library import
import os
import yaml

"""Methods in this file are used to help with the project. Any function applied frequently is best stored here"""

# Function for loading the config file
def load_config():
    # Ensuring that working directory is appropriately set
    assert os.getcwd().split('/')[-1] == 'aml_itu', 'Working directory not set to root of project. Currently working in: ' + os.getcwd()    
    
    # Loading the config file
    with open('config.yml') as file:
        config = yaml.safe_load(file)
    return config