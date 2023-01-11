# ASL(American Sign Language) Classification
### A two staged approach to Classify ASL

Domain : None
### State : Active 

## Requirements
1. Python >3.8
2. Jupyter Notebook or Lab

## Setup
- Clone the Repo
- Get the dataset
- Setup the virtual environment (Optional)
- Active the Virtual Environment
- Install the python dependency(Inside venv or on root)
- change the dataset path variable ```PATH``` with your local path.
- Run the ```ASL_dataset generator.ipynb```
- Run the ```ASL_keypoint_model_trainer.ipynb```
- Run the ```ASL_keypoint_detector.ipynb```
- Run the ```Model_tester.ipynb```

## Download the Dataset
Kaggle Link :- 
[Dataset](https://www.kaggle.com/datasets/ayuraj/asl-dataset?resource=download)

## Install The python dependency
```
pip install -r requirements.txt
```

## upstream the local repository with remote repository

```
git remote add upstream https://github.com/tirtharajsinha/ASL-Classifier.git
git fetch upstream
git checkout main
git merge upstream/main

```

## reset repo

```
git reset --hard origin/main
```

## Evaluation Reasult
### Hardware requirements
Device : Dell inspiron 3543
CPU : intel i3 5005U
GPU : Intel HD grapics integrated
RAM : Samsung 4GB DD3 SODIMM RAM
HDD : Kingstone 480GB SSD