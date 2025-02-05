# Automating Stepsizes for Decentralized Optimization and Learning with Geometric Convergence
Decentralized optimization is a promising paradigm for addressing fundamental challenges in machine learning. However, despite the unprecedented success of existing decentralized optimization and learning methods, the selection of effective stepsizes is still elusive.

We propose an approach that allows individual agents to autonomously adapt their individual stepsizes. 
The effectiveness of the proposed approach is confirmed using the following three typical machine learning applications on benchmark datasets, including logistic regression, matrix factorization, and image classification.
![Introduction](https://github.com/cziqin/Automated_Stepsizes/blob/main/figures/introduction.png)
## 🕵️ Outlines
- Installation Tutorial and Preliminaries
- Logistic Regression
- Matrix Factorization
- Training of Convolutional Neural Networks
- Discussions
- License

## 🔧 Installation Tutorial and Preliminaries

### Install Setup
1. Clone this repository

```bash
git clone https://github.com/AIcrowd/neurips2020-procgen-starter-kit.git
```

- Download and install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) (if you don't have it already)

- Create a new environment with python 3.7

```bash
conda create -n procgen -y
conda activate procgen
conda install python=3.7 -y
```

- Install dependencies

```bash
conda activate procgen
pip install ray[rllib]==0.8.6
pip install procgen
```

- Install the DL framework of you choice (tensorflow, pytorch, etc.,)
```sh
pip install -r requiremens.txt
```

### Hardware/computing resources
The experiments were conducted using a system with 32 CPU cores, 31GB of memory, and an NVIDIA GeForce RTX 4090 GPU with 24GB VRAM.

### Datasets
| Datasets | Download link | Storage Location|
| ------ | ------ | ------|
| Mushrooms | https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/ |`./mushrooms`|
| MovieLens 100k | https://grouplens.org/datasets/movielens/|`./matrix_factorization/data/`|
| CIFAR-10 | https://www.cs.toronto.edu/~kriz/cifar.html |`./Neural_networks/data/`|
| ImageNet | https://academictorrents.com/collection/imagenet-2012 |`./Neural_networks/data/`|

Ensure that each dataset is downloaded and placed in its corresponding folder before running the experiments.

## 💪 Logistic regression
This experimental code is located in the Logistic_regression folder. To run the main script, use the following command:
```
python Logistic_regression/main.py --test_num 0 --iterations 1000
```
- `--test_num`: Specifies the optimization algorithm to be used:\
`0`:Algorithm 1; `1`: Algorithm 3; `2`:Algorithm 4; `3`: DGM-BB-C; `4`: DGD.
- `--iterations`: sets the number of trianing iterations.
- To run Algorithm S1, please modify the `optimizers.py` file by setting `self.K=1` in the `Algorithm4` class:
```
class Algorithm4(Trainer):
    def __init__(self, *args, **kwargs):
        super(Algorithm4, self).__init__(*args, **kwargs)
        self.name = "Algorithm4"
        self.K = 1  # Change this from 1000 to 1 for Algorithm S1
        self.agent_y = {}
```
- The results will be saved as `.csv` files in the `./results/` directory. 
### Experimental results
![Introduction](https://github.com/cziqin/Automated_Stepsizes/blob/main/figures/mushrooms_png.png)




## 💪 Matrix factorization
The "MovieLens 100k" dataset used for this experiment is already included in the matrix_factorization folder. To run this experiment, please execute the ``mf.py`` file.

## 💪 Neural network training
### Cifar 10
The CIFAR-10 experiments used a four-layer CNN, which is provided in the file 'models.py' in the 'Neural_Network' folder.

### ImageNet
The ImageNet experiments used a ResNet-18 architecture, which is provided in the file 'resnet.py' within the 'Neural_Network' folder.

> Note:

## 🚀 Discussions
