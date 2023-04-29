# Federated Learning for distributed databases

Based on [fl-official-statistcs](https://github.com/joshua-stock/fl-official-statistics) this repo investigates further the feasability of Federated Learning for Official Statistics.

## Setup

more possible environment (e.g. local, R-Server) tba.

### Google Colab

Required:

- google account with
- connected google colab


Open a new notebook in Google Colab. This starts a new session (called runtime) with nothing installed or available in memory. S. [How to use Google Colaboratory to clone a GitHub Repository to your Google Drive? (Medium)](https://medium.com/@ashwindesilva/how-to-use-google-colaboratory-to-clone-a-github-repository-e07cf8d3d22b). Optionally a google drive can be mounted to backup a repo-version.

#### Pull this repo

```python
import os

# rm repo from gdrive
if os.path.exists("fl-official-statistics-addon"):
  %rm -r fl-official-statistics-addon

# clone
!git clone https://github.com/Olhaau/fl-official-statistics-addon

# pull (the currenct version of the repo)
!git pull
```

#### Push changes

After you have made changes to your notebook, you can commit and push them to the repository. To do so from within a Colab notebook, click File → Save a copy in GitHub. You will be prompted to add a commit message, and after you click OK, the notebook will be pushed to your repository (s. https://bebi103a.github.io/lessons/02/git_with_colab.html). 

For possible solutions from within a notebook s. [stackoverflow/how-to-push-from-colab-to-github](https://stackoverflow.com/questions/59454990/how-to-push-from-colab-to-github).

## Repo Structure

### _dev

Main development code can be found here as jupyter-notebooks. Currently available:

#### 01_insurance_prep_eda_baselines

For the medical insurance data, this notebook contains the preprocessing, EDA and baselines by classical ML-algorithms. To fastly calculate the baselines, AutoML is used, e.g. pycaret and TPOT.

#### 02_insurance_DNN

Tests the performance of deep neural networks (DNN) for a centralized (non federated approach) as another baseline. Further experiments try to improve the performance, stablize the training and investigates regional differences in the datasets.

#### 03_insurance_federated

tba: Federated Learning for the medical insurance data.

### Other folders

- **output:** processed data, generated plots and reports
- **archive:** some raw notebooks. Relevant code will in the future be transfered to _dev.
- **doc:** contains notes on meetings. 


### original_work

This folder contains the original repository. It is divided into three main parts. These can be found in the contained top folders.

#### PM2.5 prediction for Beijing

The pm-beijing folder contains data with weather and air pollution data and can be used to predict the PM2.5 concentration in the air. The folder contains the following subfolders and files:
* data: The pm25 beijing dataset
* models: The trained models are saved here, it contains already trained models
* tensorboard-logs: The log files used by tensorboard
* pm25_beijing.py: The python module containing all the functions used in the notebooks
* pm25_central_model.ipynb: The single model, trained on all the data
* pm25_cross_validation.ipynb: The above model but trained with cross validation
* pm25_feature_comparison.ipynb: A simple model to compare how time and wind direction affect the prediction
* pm25_federated.ipynb: A federated model for PM2.5 prediction
* pm25_federated_cross_val.ipynb: The above federated model with cross validation
* pm25_hyperparametersearch.ipynb: A hyperparametersearch for optimising the model
* pm25_local_models.ipynb: A local model for each station is trained here

#### LTE data from Umlaut

This folder contains everything related to Umlaut and their data. As the original data from Umlaut is not accessible it contains only short dummy data to test the models.
* data: The data files, both files hold the same kind of information, however be aware that the name of the features differ slightly
* lte_federated.ipynb: The federated model for the LTE data and centralised models for comparison reasons
* lte_hyperparametersearch.ipynb: A hyperparametersearch for different models
* umlaut_lte.py: A python module containing some functions used by the notebooks

#### Medical insurance data

Everything concerning the medical insurance data is stored here. This dataset is used to calculate charges a person has to pay and contains data as age, sex, BMI, smoker, number of children and region.
* data: The medical insurance data
* med_insurance.ipynb: Some general data processing steps and a centralised model for medical insurance charge prediction
* med_insurance_federated.ipynb: The federated model for medical insurance charge prediction

## Further Resources

- [Accountable Federated Machine Learning in Government](https://link.springer.com/chapter/10.1007/978-3-030-82824-0_10)
- [EU TechSonar](https://edps.europa.eu/press-publications/publications/techsonar/federated-learning_en)

