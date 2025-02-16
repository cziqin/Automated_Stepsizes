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
1. Clone this [repository](https://github.com/cziqin/Automated_Stepsizes/tree/main)
2. Download and install [Anaconda](https://www.anaconda.com) (if you don't have it already)
3. Create a new conda environment with python 3.12
```bash
conda create -n autostep python=3.12
conda activate autostep
```
4. Install any additional packages you need in this environment using conda or pip (tensorflow, pytorch, etc.,)
```sh
pip install -r requiremens.txt
```

### Hardware/computing resources
The experiments were conducted using a system with 32 CPU cores, 31GB of memory, and an NVIDIA GeForce RTX 4090 GPU with 24GB VRAM.

### Repository Structure

```
├── Logistic_regression                 # Directory to implement a logistic regression classification problem
│   ├── results                         # .csv files for experimental results
│   ├── loss_function.py                # Defines the loss function 
│   ├── main.py                         # Entry point
│   ├── matrix.py                       # generates communication matrix and excuctes Subroutine 1
│   ├── mushrooms                       # datasets (DO NOT EDIT)
│   ├── optimizer.py                    # Optimization algorithms
│   └── train.py                        # Training script for model training and evaluation
├── Matirx_factorization
│   ├── __init__.py
│   ├── framestack.py                   
│   ├── procgen_env_wrapper.py          
├── Neural_networks                         
│   ├── impala-baseline.yaml            
│   ├── procgen-starter-example.yaml    
│   └── random-policy.yaml              
├── LICENSE                             # License file
├── README.md                                             
└── requirements.txt                    # List of dependencies required to run the project             
```

### Datasets
| Datasets | Download link | Storage Location|
| ------ | ------ | ------|
| Mushrooms | https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/ |`./Logistic_regression/`|
| MovieLens 100k | https://grouplens.org/datasets/movielens/|`./matrix_factorization/data/`|
| CIFAR-10 | https://www.cs.toronto.edu/~kriz/cifar.html |`./Neural_networks/data/`|
| ImageNet | https://academictorrents.com/collection/imagenet-2012 |`./Neural_networks/data/`|

Ensure that each dataset is downloaded and placed in its corresponding folder before running the experiments.

## 💪 Logistic regression
1. You can use the following command to execute the logistic regression model:
```
python .\main.py --test_num 0 --iterations 1000
```
![Mushroom](https://github.com/cziqin/Automated_Stepsizes/blob/main/figures/mushroom.gif)
- `--test_num`: Specifies the optimization algorithm to be trained: `0`:Algorithm 1; `1`: Algorithm 2; `2`: DGM-BB-C; `3`: DGD.
- `--iterations`: sets the number of trianing iterations.
2. To execute Algorithm 2 with a desired number of inner-consensus-loop iterations $K_0$ (e.g., $K_{0}=10$), you can reset the parameter  `K_LOOP` (e.g., `K_LOOP=10`) in the [`matrix.py`](https://github.com/cziqin/Automated_Stepsizes/blob/main/Logistic_regression/matrix.py) file. Please run:
```
(Get-Content matrix.py) -replace 'K_LOOP = 1', 'K_LOOP = 10' | Set-Content matrix.py
python .\main.py --test_num 0 --iterations 1000
```
  
3. To execute Algorithm 3 with a desired number of asynchronous-parallel-update iterations $Q_0$ (e.g., $Q_{0}=10$), you can first reset the parameter  `CONST_Q` (e.g., `CONST_Q=10`) in the [`matrix.py`](https://github.com/cziqin/Automated_Stepsizes/blob/main/Logistic_regression/matrix.py) file, and then execute Algorithm 1. Please run:
```
(Get-Content matrix.py) -replace 'CONST_Q = 1', 'CONST_Q = 30' | Set-Content matrix.py
python .\main.py --test_num 0 --iterations 1000
```

4. All experimental results (including loss, wallclock time, average stepsizes) will be automously saved as `.csv` files in the `./Logistic_regression/results` directory.

> Note: Parameter `K_LOOP` represents the number of inner-consensus-loop ietrations in Algorithm 2 and DGM-BB-C; Parameter `CONST_Q` represents the number of asynchronous-parallel-update iterations in Algorithm 3.
### Experimental results
![Fig3](https://github.com/cziqin/Automated_Stepsizes/blob/main/figures/mushrooms_png.png)

- Fig. A (and its zoomed-in view of iterations 40 to 160) shows the loss evoluation of Algorithm 1, Algorithm S1, Algorithm 3 with Q=5, Algorithm 4 with K=10, DGM-BB-C with K=10, and DGD, respectively.
- Fig. B compares the communication rounds used by our Algorithm 1 (synchronous updates) and Algorithm 3 (asynchronous parallel updates) under different numbers of asynchronous-parallel-update iterations.
- Fig. C presents a comparison of the average stepsize of five agents across the six algorithms.
- Fig. D shows the median, first and third quartiles, and the minimum and maximum values of the average stepsize in the six algorithms.
- Fig. E compares the average, minimum, and maximum differences in stepsizes between pairs of algorithms.
- Fig. F shows comparision results of Algorithm 1 with Algorithm S1, Algorithm 3 with Q=5, Algorithm 4 with K=10, DGM-BB-C with K=10, and DGD in terms of wallclock time, respectively.

## 💪 Matrix factorization
The "MovieLens 100k" dataset used for this experiment is already included in the matrix_factorization folder. To run this experiment, please execute the ``mf.py`` file.

## 💪 Neural network training
### Cifar 10
The CIFAR-10 experiments used a four-layer CNN, which is provided in the file 'models.py' in the 'Neural_Network' folder.

### ImageNet
The ImageNet experiments used a ResNet-18 architecture, which is provided in the file 'resnet.py' within the 'Neural_Network' folder.

> Note:

## 🚀 Discussions

## License

## Authors
- [Ziqin Chen](https://scholar.google.com/citations?user=i-IM2rIAAAAJ&hl=zh-CN)
- [Yongqiang Wang](https://scholar.google.com/citations?hl=zh-CN&user=shSZpGUAAAAJ)
