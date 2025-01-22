# Automating Stepsizes for Decentralized Optimization and Learning with Geometric Convergence
This project contains three machine learning experiments: logistic regression, matrix factorization, and neural network training. 
## Outlines
- Installation Tutorial
- Logistic regression
- Matrix factorization
- Training of convolutional neural networks
- Discussions
- License

## Installation Tutorial
### Environment Requirements
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

### Install Setup
```sh
pip 
npm i
node app
```

### Hardware/computing resources
The experiments were conducted using a system with 32 CPU cores, 31GB of memory, and an NVIDIA GeForce RTX 4090 GPU with 24GB VRAM.

### Datasets
| Datasets | Download link |
| ------ | ------ |
| Mushrooms | https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/ |
| MovieLens 100k | https://grouplens.org/datasets/movielens/|
| CIFAR-10 | https://www.cs.toronto.edu/~kriz/cifar.html |
| ImageNet | https://academictorrents.com/collection/imagenet-2012 |


## Logistic regression
The "mushrooms" dataset used for this experiment is already included in the logistic_regression folder. To run this experiment, please execute the ``train.py`` file.

## Matrix factorization
The "MovieLens 100k" dataset used for this experiment is already included in the matrix_factorization folder. To run this experiment, please execute the ``mf.py`` file.

## Neural network training
### Cifar 10
The CIFAR-10 experiments used a four-layer CNN, which is provided in the file 'models.py' in the 'Neural_Network' folder.

### ImageNet
The ImageNet experiments used a ResNet-18 architecture, which is provided in the file 'resnet.py' within the 'Neural_Network' folder.

> Note:







