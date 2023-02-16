# ASL (American Sign Language) Classification
> A two staged approach to Classify Hand sign

<img src="art.jpg" alt="" style="border-radius:20px; opacity:8">

- Domain : None

- State : Active 

## Aim
The aim is to develope a method to classify hand sign from image efficiently and building a ***Realtime sign detector Application***. Classical Hand sign detector models train directly from images and I have discovered that it negitively affects the realtime detection accuracy.

<br>These are factors are :- 
- Hand side [Left/Right]
- Skin color and birthmarks on hand
- Hand distance from camara
- Camara quality
- Hand angel
- Rapid movement
- Ambient lighting
- Background noise, color, movement

We have tried  to eliminate this limitation with a different 2 staged approach. 

## Requirements
1. Python >3.8
2. Jupyter Notebook or Lab
3. git

## Setup
- Clone the Repo
```
git clone https://github.com/tirtharajsinha/ASL-Classifier.git
```
- Get the [dataset](https://www.kaggle.com/datasets/ayuraj/asl-dataset?resource=download)
- Setup and active the virtual environment (Optional)
- Install the python dependency(Inside venv or on root)
```
pip install -r requirements.txt
```
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
pip install -r requirements.txt --user
```

## Run the realtime ASL detection
```
python trackOnCam.py
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

### Hardware (Tested)
- Device : Dell inspiron 3543
- CPU : intel i3 5005U
- GPU : Intel HD grapics integrated
- RAM : Samsung 4GB DD3 SODIMM RAM
- HDD : Kingstone 480GB SSD

### Software (Tested)
- OS : Windows 10 22H2 / Linux mint 20.3
- Language : Python3.9
- Package distributor : Anaconda
- IDE/interface : Jupyter notebook

### Result
- CSV dataset Generate time : 292 Seconds
- Training time : 83.47s
- Accuracy : 92%
- Detectction time for One image : 62ms


<hr>
<p style="text-align:right; font-size:20px">By Tirtharaj Sinha</p>

