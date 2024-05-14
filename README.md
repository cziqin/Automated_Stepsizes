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

### Running the Experiments

#### 1. Logistic Regression
The main program for the logistic regression experiment is train.py, located in the logistic_regression folder. 

The "Mushrooms" dataset used in this experiment was downloaded from the public repository available at https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/

#### 2. Matrix Factorization
The main program for the matrix factorization experiment is mf.py, located in the matrix_factorization folder. 

The ``MovieLens 100K" dataset used in this experiment was downloaded from the public repository available at https://grouplens.org/datasets/movielens/ 

#### 3. Neural Network Training
The main program for the neural network training experiment is main.py, located in the neural_network folder. 

The "MNIST" dataset and the "CIFAR-10" dataset will be automatically loaded. They are sourced from the public repositories at http://yann.lecun.com/exdb/mnist/ and https://www.cs.toronto.edu/~kriz/cifar.html, respectively.

