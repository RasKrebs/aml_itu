# Focus Fumblers

### Classifying Distracted Drivers
A repository for everything related to the exam project of Advanced Machine Learning at IT University of Copenhagen

![Distracted Drivers Gif](https://storage.googleapis.com/kaggle-media/competitions/kaggle/5048/media/output_DEb8oT.gif)

## Project folders and structured
The main files and subfolders in this repository, are structured as follows:

aml_itu: Main project
- `predict.py`: A prediction pipeline script, that will access webcamera and predict based on this. Description on how to use this is provided at the bottom.
- `test.ipynb`: Notebook, in which all models are tested on the test set. Additionally, a GradCam analysis is carried out.
- utils/: Folder for helper functions and other project-specific modules
- notebooks/: Folder that contains all notebooks used for training of our models
    - archive/: Archived models notebooks such as MobileNet and Mnas.
    - `cv_result_explorer.ipynb`: Explores results from cross validation test
    - `data_exploration.ipynb`: Explores and visualize the data
    - `data_exploration.ipynb`: Explores and visualize the data
- outputs/: Folder for outputs, mainly models in `.pt` format, but also results from CV tests.

state-farm-distracted-driver-detection: dataset folder 
- `driver_imgs_list.csv`: Image metadata dataset, containing information about classes, subjects, files etc.
- imgs/: 
    - train/: Subfolder with training images
        - c0/: Training images in class c0
        ....
        - c9/: Training images in class c9



## Setting up project from scratch
To setup the project properly, data must first be downloaded. The data used for this project is `Distracted Driver Detection` dataset released in relation to a [State Farm Kaggle Competition](https://www.kaggle.com/competitions/state-farm-distracted-driver-detection/overview). To download the data, you will first need to setup kaggle CLI, extract the necessary API credentials and approve competition rules. If you have done this previously, skip to step see step 3, if not, follow the steps below 

### 1. Installing Kaggle CLI
Paste below command in your command line
```zsh
pip install kaggle
```

### 2. Downloading API Credentials
To use the Kaggle API, a Kaggle account is required, so signup via [Kaggle](https://www.kaggle.com). When accounted is created, go to the `Account` tab of your user profile and select 'Create API Token'. This will trigger the download of kaggle.json, a file containing your API credentials. Place this file in the location `~/.kaggle/kaggle.json` (on Windows in the location `C:\Users\<Windows-username>\.kaggle\kaggle.json` - you can check the exact location, sans drive, with echo %HOMEPATH%). 

### 3. Approving Competition Rules
To download the data from the project, you will need to approve competition rules. Go to the rules tab under the (comptetion)[https://www.kaggle.com/competitions/state-farm-distracted-driver-detection/overview] and approve the rules. When this is done, a green checkmark should appear under the tab, indicating you have acceted the rules.

### 4. Downloading Data
Paste below command in your command line
```zsh
kaggle competitions download -c state-farm-distracted-driver-detection
```


### 5. Cloning Repo
When steps 1-4 are done, in the same folder where your now download `state-farm-distracted-driver-detection` dataset is located, clone this repo
```zsh
git clone https://github.com/RasKrebs/aml_itu.git
```

## Inference Pipeline `predict.py`
To apply and run the model, use the associated predict file. 
```sh
$ python predict.py 
    --model: Specifies which model to use. Either tinyvgg, efficientnetb0, efficientnetb1, vgg16 or resnet
    --size Image size for transformation: Either X_SMALL, used for TinyVGG or L_SQUARED for all ohter models
    --seconds Number of seconds to run inference: Default is 10
    --weighted_frames Determines how many frames to use weigh and use for prediction. Defeault is 10
    --source webcam or filepath to picture. If file path, seconds and weighted frames are irrelvant. Default is webcam
    --device gpu or cpu. If gpu, file will automatically determine if cuda or mps is available and use that. Default is GPU.
```
