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
│   ├── mushrooms                       # the mushrooms datasets
│   ├── optimizer.py                    # Optimization algorithms
│   └── train.py                        # Training script for model training and evaluation
├── Matirx_factorization
│   ├── main.py                         # Entry point
│   ├── matrix.py                       # generates communication matrix and excuctes Subroutine 1
│   ├── optimizer.py                    # Optimization algorithms
│   └── u.data                          # the MovieLens 100k dataset
├── Neural_networks                         
│   ├── datadeal.py                     # Splits the downloaded ImageNet dataset into training and test sets          
│   ├── main.py                         # Entry point
│   ├── matrix.py                       # generates communication matrix and excuctes Subroutine 1
│   ├── models.py                       # the model used in CNN training on the Cifar-10 dataset
│   ├── ops.py                          # Optimization algorithms
│   ├── resnet.py                       # the model used in CNN training on the ImageNet dataset   
│   └── train.py                        # Training script for model training and evaluation
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

Ensure that each dataset is downloaded and placed in its corresponding directory before running the experiments.

## 💪 Logistic regression
1. You can use the following command to execute Algorithm 1 for the logistic regression task:
```
python .\Logistic_regression\main.py --test_num 0 --iterations 1000
```
> Note: Here, `.\Logistic_regression\main.py` is a relative path, meaning the script should be executed from the directory containing the `Logistic_regression` directory.

![Mushroom](https://github.com/cziqin/Automated_Stepsizes/blob/main/figures/mushrooms_execution.gif)
- `--test_num`: Specifies the optimization algorithm to be trained: `0`:Algorithm 1; `1`: Algorithm 2; `2`: DGM-BB-C; `3`: DGD.
- `--iterations`: sets the number of trianing iterations.
2. To execute Algorithm 2 with a desired number of inner-consensus-loop iterations $K$ (e.g., $K=10$), you can reset the parameter  `K_LOOP` (e.g., `K_LOOP=10`) in the [`matrix.py`](https://github.com/cziqin/Automated_Stepsizes/blob/main/Logistic_regression/matrix.py) file. For example, you can run the following command in the Windows PowerShell:
```
(Get-Content matrix.py) -replace 'K_LOOP = 1', 'K_LOOP = 10' | Set-Content matrix.py
python .\main.py --test_num 0 --iterations 1000
```
  
3. To execute Algorithm 3 with a desired number of asynchronous-parallel-update iterations $Q$ (e.g., $Q=10$), you can first reset the parameter  `CONST_Q` (e.g., `CONST_Q=10`) in the [`matrix.py`](https://github.com/cziqin/Automated_Stepsizes/blob/main/Logistic_regression/matrix.py) file, and then execute Algorithm 1. For example, you can run the following command in the Windows PowerShell:
```
(Get-Content matrix.py) -replace 'CONST_Q = 1', 'CONST_Q = 30' | Set-Content matrix.py
python .\main.py --test_num 0 --iterations 1000
```
> Note: Parameter `K_LOOP` represents the number of inner-consensus-loop ietrations in Algorithm 2 and DGM-BB-C; Parameter `CONST_Q` represents the number of asynchronous-parallel-update iterations in Algorithm 3.

4. In this experiment, we set the stepsize $\eta=1/L_{\max}=0.0351132$ for DGD, which follows the guideline in [42]. The stepsizes of Algorithm 1, Algorithm 2, and DGM-BB-C are automatively adjusted without requiring any mannual tuning.  

### Experimental results
<div align="center">
  <img src="https://github.com/cziqin/Automated_Stepsizes/blob/main/figures/mushrooms.png" alt="Fig3" width="900">
</div>

- Fig. a shows the loss evolution of Algorithm 1, Algorithm 2 with K=1, Algorithm 2 with K=10, Algorithm 3 with Q=10, DGM-BB-C with K=1, and DGD over iterations, respectively.
- Fig. b shows the average stepsize of five agents across the comparison algorithms over iterations.
- Fig. c shows the median, first and third quartiles, and the 5th to 95th percentiles of the average stepsize in the six algorithms.
- Fig. d shows the comparision results of Algorithm 1 with Algorithm 2, Algorithm 3, DGM-BB-C, and DGD in terms of wallclock time, respectively.
- Fig. e shows the comparision results of Algorithm 1 with Algorithm 2, Algorithm 3, DGM-BB-C, and DGD in terms of communication rounds, respectively.
- Fig. f shows the comparision results of Algorithm 1 (synchronous parallel updates) with Algorithm 3 (asynchronous parallel updates) under different numbers of asynchronous-parallel-update iterations.

> Note: All experimental results (including loss, wallclock time, average stepsizes) will be automously saved as `.csv` files in the `./Logistic_regression/results` directory.
## 💪 Matrix factorization
1. You can use the following command to execute Algorithm 1 for the matrix factorization task:
```
python .\Matrix_factorization\main.py --test_num 0 --iterations 1000
```
![Matrix](https://github.com/cziqin/Automated_Stepsizes/blob/main/figures/Matrix_factorization_execution.gif)
- `--test_num`: Specifies the optimization algorithm to be trained: `0`:Algorithm 1; `1`: Algorithm 2; `2`: DGM-BB-C; `3`: DGD.
- `--iterations`: sets the number of trianing iterations.

2. To execute Algorithm 2 with a desired number of inner-consensus-loop iterations $K$ (e.g., $K=10$), you can reset the parameter  `K_LOOP` (e.g., `K_LOOP=10`) in the [`matrix.py`](https://github.com/cziqin/Automated_Stepsizes/blob/main/Matrix_factorization/matrix.py) file. For example, you can run the following command in the Windows PowerShell:
```
(Get-Content matrix.py) -replace 'K_LOOP = 1', 'K_LOOP = 10' | Set-Content matrix.py
python .\Matrix_factorization\main.py --test_num 1 --iterations 1000
```
  
3. To execute Algorithm 3 with a desired number of asynchronous-parallel-update iterations $Q$ (e.g., $Q=10$), you can first reset the parameter  `CONST_Q` (e.g., `CONST_Q=10`) in the [`matrix.py`](https://github.com/cziqin/Automated_Stepsizes/blob/main/Matrix_factorization/matrix.py) file, and then execute Algorithm 1. For example, you can run the following command in the Windows PowerShell:
```
(Get-Content matrix.py) -replace 'CONST_Q = 1', 'CONST_Q = 30' | Set-Content matrix.py
python .\Matrix_factorization\main.py --test_num 0 --iterations 1000
```

4. In this experiment, we set the stepsize $\eta=10^{-4}$ for the DGD, since it is the best suboptimal step size we found based on the following tuning results after 200 iterations:
   
<table>
  <tr> <th rowspan="2">Algorithms</th> <th colspan="9">Stepsizes</th>
  </tr>
  <tr> <th>$10^{-8}$</th>   <th>$10^{-7}$</th>   <th>$10^{-6}$</th> <th>$10^{-5}$</th>   <th>$10^{-4}$</th>   <th>$10^{-3}$</th>
    <th>$10^{-2}$</th>   <th>$10^{-1}$</th>   <th>$10^{0}$</th>
  </tr>
  <tr>
    <td>DGD</td> <td>5.673 &pm; 0.01</td>  <td>5.667 &pm; 0.01</td>  <td>5.614 &pm; 0.01</td>  <td>5.330 &pm; 0.01</td>
    <td>5.135 &pm; 0.01</td>  <td>nan</td>  <td>nan</td>  <td>nan</td>  <td>nan</td>
  </tr>
  <tr>
    <td>Algorithm 1 </td>  <td colspan="9">5.095 &pm; 0.01</td>
  </tr>
</table>

### Experimental results
<div align="center">
  <img src="https://github.com/cziqin/Automated_Stepsizes/blob/main/figures/matrix_factorization.png" alt="Fig4" width="900">
</div>

- Fig. a shows the loss evolution of Algorithm 1, Algorithm 2 with K=1, Algorithm 2 with K=10, Algorithm 3 with Q=10, DGM-BB-C with K=1, and DGD over iterations, respectively.
- Fig. b shows the average stepsize of five agents across the comparison algorithms over iterations.
- Fig. c shows the median, first and third quartiles, and the 5th to 95th percentiles of the average stepsize in the six algorithms.
- Fig. d shows the comparision results of Algorithm 1 with Algorithm 2, Algorithm 3, DGM-BB-C, and DGD in terms of wallclock time, respectively.
- Fig. e shows the comparision results of Algorithm 1 with Algorithm 2, Algorithm 3, DGM-BB-C, and DGD in terms of communication rounds, respectively.
- Fig. f shows the comparision results of Algorithm 1 (synchronous parallel updates) with Algorithm 3 (asynchronous parallel updates) under different numbers of asynchronous-parallel-update iterations.

> Note: All experimental results (including loss, wallclock time, average stepsizes) will be automously saved as `.csv` files in the `./Matrix_factorization/results` directory.

## 💪 Neural network training
### Cifar 10
1. You can use the following command to execute Algorithm 1 for the matrix factorization task:
```
python .\Neural_networks\main.py --test_num 0 --epochs 70 --batch_size 128 --dataset 'cifar10'
```

>Note: Before running the script, please ensure that the CIFAR-10 dataset has been downloaded and placed in the `./Neural_networks/data` directory.

- `--test_num`: Specifies the optimization algorithm to be trained: `0`:Algorithm 3; `1`: DADAM; `2`: DAMSGrad; `3`: DSGD-N; `4`: ATC-DIGing; `5`: DSGD.
- `--epochs`: sets the number of trianing epochs.
- `batch_size`: sets the batch size for training.
- `dataset`: Specifies the dataset to be used for training. The default option is 'cifar10'.

2. To execute Algorithm 3 with a desired number of asynchronous-parallel-update iterations $Q$ (e.g., $Q=10$), you need to first set the parameter  `CONST_Q` (e.g., `CONST_Q=10`) in the [`matrix.py`](https://github.com/cziqin/Automated_Stepsizes/blob/main/Neural_networks/matrix.py) file, and then execute Algorithm 3. For example, you can run the following command in the Windows PowerShell:
```
(Get-Content matrix.py) -replace 'CONST_Q = 1', 'CONST_Q = 10' | Set-Content matrix.py
python .\Neural_networks\main.py --test_num 0 --epochs 70 --batch_size 128 --dataset 'cifar10'
```

3. To specify the print interval (e.g., printing the training loss, test accuracy, and average stepsize every 10 iterations), you need to first update the parameter `SPECIFIC_LOG_INTERVAL` (e.g., `SPECIFIC_LOG_INTERVAL=10`) in the [`matrix.py`](https://github.com/cziqin/Automated_Stepsizes/blob/main/Neural_networks/matrix.py) file, and then execute Algorithm 3. For example, you can run the following command in Windows PowerShell:
```
(Get-Content matrix.py) -replace 'SPECIFIC_LOG_INTERVAL = 70', 'SPECIFIC_LOG_INTERVAL = 10' | Set-Content matrix.py
python .\Neural_networks\main.py --test_num 0 --epochs 70 --batch_size 128 --dataset 'cifar10'
```

4. To specify the random seed used in training (e.g., setting seed=42), you can first update the parameter `SEED` (e.g., `SEED = 42`) in the [`matrix.py`](https://github.com/cziqin/Automated_Stepsizes/blob/main/Neural_networks/matrix.py) file, and then execute Algorithm 3. The default seed is 42.


### Experimental results
<div align="center">
  <img src="https://github.com/cziqin/Automated_Stepsizes/blob/main/figures/cifar10.png" alt="Fig5" width="900">
</div>

- Fig. a shows the loss evolution of Algorithm 1, Algorithm 2 with K=1, Algorithm 2 with K=10, Algorithm 3 with Q=10, DGM-BB-C with K=1, and DGD over iterations, respectively.
- Fig. b shows the average stepsize of five agents across the comparison algorithms over iterations.
- Fig. c shows the median, first and third quartiles, and the 5th to 95th percentiles of the average stepsize in the six algorithms.
- Fig. d shows the comparision results of Algorithm 1 with Algorithm 2, Algorithm 3, DGM-BB-C, and DGD in terms of wallclock time, respectively.
- Fig. e shows the comparision results of Algorithm 1 with Algorithm 2, Algorithm 3, DGM-BB-C, and DGD in terms of communication rounds, respectively.
- Fig. f shows the comparision results of Algorithm 1 (synchronous parallel updates) with Algorithm 3 (asynchronous parallel updates) under different numbers of asynchronous-parallel-update iterations.

> Note: All experimental results (including loss, wallclock time, average stepsizes) will be automously saved as `.csv` files in the `./Matrix_factorization/results` directory.



### ImageNet
The ImageNet experiments used a ResNet-18 architecture, which is provided in the file 'resnet.py' within the 'Neural_Network' directory.

> Note:

## 🚀 Discussions

## License

## Authors
- [Ziqin Chen](https://scholar.google.com/citations?user=i-IM2rIAAAAJ&hl=zh-CN)
- [Yongqiang Wang](https://scholar.google.com/citations?hl=zh-CN&user=shSZpGUAAAAJ)
