# Focus Fumblers
### Classifying Distracted Drivers
A repository for everything related to the exam project of Advanced Machine Learning at IT University of Copenhagen
___
## Setting up project
To setup the project properly, data must first be downloaded. The data used for this project is `Distracted Driver Detection` dataset released in relation to a [State Farm Kaggle Competition](https://www.kaggle.com/competitions/state-farm-distracted-driver-detection/overview). To download the data, you will first need to setup kaggle CLI, extract the necessary API credentials and approve competition rules. If you have done this previously, skip to step see step 3, if not, follow the steps below 

### 1: Installing Kaggle CLI
Paste below command in your command line
```zsh
pip install kaggle
```

### 2: Downloading API Credentials
To use the Kaggle API, a Kaggle account is required, so signup via [Kaggle](https://www.kaggle.com). When accounted is created, go to the `Account` tab of your user profile and select 'Create API Token'. This will trigger the download of kaggle.json, a file containing your API credentials. Place this file in the location `~/.kaggle/kaggle.json` (on Windows in the location `C:\Users\<Windows-username>\.kaggle\kaggle.json` - you can check the exact location, sans drive, with echo %HOMEPATH%). 

### 3: Approving Competition Rules
To download the data from the project, you will need to approve competition rules. Go to the rules tab under the (comptetion)[https://www.kaggle.com/competitions/state-farm-distracted-driver-detection/overview] and approve the rules. When this is done, a green checkmark should appear under the tab, indicating you have acceted the rules.

### 4: Downloading Data
Paste below command in your command line
```zsh
kaggle competitions download -c state-farm-distracted-driver-detection
```


### 5. Cloning Repo
When steps 1-4 are done, in the same folder where your now download `state-farm-distracted-driver-detection` dataset is located, clone this repo
```zsh
git clone https://github.com/RasKrebs/aml_itu.git
```
