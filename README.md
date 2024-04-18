# Autoencoder Graph Classfication

![image](https://raw.githubusercontent.com/wuyiulin/AGC/main/img/AGC_model.png)

## Introduction

#### This is a two step image classification solution.<br/> First I use unsuperviser method trained a AutoEncoder model,<br/>in second step I training a Classifier to complete this task.

## My environment
```bash=
 Nvidia RTX 4060
 cuda version 12.2
 PyTorch version 1.81+cu111
```
#### Although I using Python3.9, but i guess if you can successful install PyTorch 1.81+cu111, this repo should be work for you.</br>Because I didn't use any unique function of Python3.9 or Syntactic sugar ：）

## Dataset

#### Here I take the [Dogs vs. Cats](https://www.kaggle.com/competitions/dogs-vs-cats-redux-kernels-edition/overview) dataset from Kaggle as an example demonstration.

#### I also split dataset by ratio 8:2 (train : test).

## Init your environment

#### Recommond use conda environment
```bash=
conda create --name AGC python=3.9
conda activate AGC
```
#### Install requirement.txt
```bash=
git clone https://github.com/wuyiulin/AGC.git
cd AGC
pip install -r requirements.txt
```

#### Prepare your dataset and follow this architecture:

```bash=
├── Train dataset 
│   └── fold(include all class training image, w/o label.)
└── Test  dataset
    ├── Class 1 image fold(only include class1 images.)
    :
    :
    :
    └── Class N image fold(only include classN images.)

```

#### Replace your dataset path in AutoEncoder.py and Classifier.py


## Start
### Step1-1: Training AutoEncoder
```bash=
python AutoEncoder.py train
```
### Step1-2: Check AutoEncoder training loss
```bash=
python VisionLoss.py
```
![image](https://raw.githubusercontent.com/wuyiulin/AGC/main/img/AutoEncoder%20Loss.png)

#### You can check training loss here, if loss curve too ridiculous to use?</br>Kick this model weight to the Death Star, it deserved.

### Step1-2: Check AutoEncoder training loss
```bash=
python AutoEncoder.py vis
```
![image](https://raw.githubusercontent.com/wuyiulin/AGC/main/img/CatandDog.png)

#### Sometimes we can't only depend loss curve to judge a model,</br>because all that glitters is not gold.

#### So you must check out those lovely kitties<3</br>If, unfortunately, they generated something ugly, you know what to do ^Q^

### Step2-1: Training Classifier
```bash=
python Classifier.py train
```

### Step2-2: Test Classifier
```bash=
python Classifier.py test
```

```bash=
Test Loss: 2.0173, Accuracy: 74.48%
```

#### You can check final accuracy of this task,</br>it seems not a perfect score in [Dogs vs. Cats](https://www.kaggle.com/competitions/dogs-vs-cats-redux-kernels-edition/overview) dataset.
#### But in another dataset with natural lighting interference and multiple noises, this model achieved very high scores.

```bash=
Test Loss: 0.0009, Accuracy: 99.94%
```

#### So I have full confidence that this model has certain usability in optical inspection in industrial or simpler environmental settings.&#x1F624;

#### <img src="https://pic.sopili.net/pub/emoji/twitter/2/72x72/26a0.png" width=20 height=20> For my convenience, the program will automatically clear</br> the /checkpoints and /log directories where model weights and loss records are stored in step 1-1.</br> Please refrain from storing any valuable items there </br>(e.g. exceptionally well-trained model weights). <3 

#### If you wish to modify this behavior, please go to the train function in AutoEncoder.py.

## Contact
Further information please contact me.

wuyiulin@gmail.com