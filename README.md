# Automating Stepsizes for Decentralized Optimization and Learning with Geometric Convergence

This project contains three machine learning experiments: logistic regression, matrix factorization, and neural network training. 
### Requirements
Please ensure that the following packages are installed:
```
torch==2.0.0+cu117
torchaudio==2.0.0+cu117
torchvision==0.15.0+cu117
python==3.11.3
numpy==1.26.4
scipy==1.12.0
pandas==2.2.1
sklearn==1.0.2
matplotlib==3.8.3
```

### Running the experiments

#### 1. Logistic regression
The "mushrooms" dataset used for this experiment is already included in the logistic_regression folder. To run this experiment, please execute the ``train.py`` file.

#### 2. Matrix factorization
The "MovieLens 100k" dataset used for this experiment is already included in the matrix_factorization folder. To run this experiment, please execute the ``mf.py`` file.

#### 3. Neural network training

The "CIFAR-10" dataset used for this experiment will automatically download when you run the ``main.py`` file.

The "ImageNet" dataset is available at https://academictorrents.com/collection/imagenet-2012
