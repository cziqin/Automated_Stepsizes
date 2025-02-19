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
├── Logistic_regression                 # directory to implement a logistic regression classification problem
│   ├── results                         # .csv files for experimental results
│   ├── loss_function.py                # defines the loss function 
│   ├── main.py                         # entry point
│   ├── matrix.py                       # generates communication matrix and excuctes Subroutine 1
│   ├── mushrooms                       # the mushrooms datasets
│   ├── optimizer.py                    # optimization algorithms
│   └── train.py                        # training script for model training and evaluation
├── Matirx_factorization
│   ├── main.py                         # entry point
│   ├── matrix.py                       # generates communication matrix and excuctes Subroutine 1
│   ├── optimizer.py                    # optimization algorithms
│   └── u.data                          # the MovieLens 100k dataset
├── Neural_networks                         
│   ├── datadeal.py                     # splits the downloaded ImageNet dataset into training and test sets          
│   ├── main.py                         # entry point
│   ├── matrix.py                       # generates communication matrix and excuctes Subroutine 1
│   ├── models.py                       # the model used in CNN training on the Cifar-10 dataset
│   ├── ops.py                          # optimization algorithms
│   ├── resnet.py                       # the model used in CNN training on the ImageNet dataset   
│   └── train.py                        # training script for model training and evaluation
├── LICENSE                             # License file
├── README.md                                             
└── requirements.txt                    # list of dependencies required to run the project             
```

### Datasets
| Datasets | Download link | Storage Location|
| ------ | ------ | ------|
| Mushrooms | https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/ |`./Logistic_regression/`|
| MovieLens 100k | https://grouplens.org/datasets/movielens/|`./matrix_factorization/`|
| CIFAR-10 | https://www.cs.toronto.edu/~kriz/cifar.html |`./Neural_networks/data/`|
| ImageNet | https://academictorrents.com/collection/imagenet-2012 |`./Neural_networks/data/imagenet/`|

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
(Get-Content matrix.py) -replace 'CONST_Q = 1', 'CONST_Q = 10' | Set-Content matrix.py
python .\main.py --test_num 0 --iterations 1000
```
> Note: Parameter `K_LOOP` represents the number of inner-consensus-loop ietrations in Algorithm 2 and DGM-BB-C; Parameter `CONST_Q` represents the number of asynchronous-parallel-update iterations in Algorithm 3.

4. In this experiment, following the guideline in [42], we set the stepsize for DGD as $\eta=1/L_{\max}\approx 0.351131704$.

5. Our tuning-free stepsize code in the `optimizer.py` file is given as follows:
<div align="center">
  <img src="https://github.com/cziqin/Automated_Stepsizes/blob/main/figures/mush_step.png" alt="Figstep" width="900">
</div>

### Experimental results
We compared our Algorithm 1, Algorithm 2, and Algorithm 3 with a baseline algorithm, i.e., DGD in [48] using a constant stepsize and the decentralized adaptive algorithm using Barzilai-Borwein stepsize in [42] (called DGM-BB-C).

- Algorithm 1 is an automated stepsize approach for decentralized optimization and learning with using a finite-time consensus strategy.
- Algorithm 2 is a variant of Algorithm 1, which uses a standard consensus protocol [63] instead of a finite-time consensus strategy in the inner-consensus-loop iterations, thereby introducing consensus errors at each outer iteration. The number of inner-consensus-loops in Algorithm 2 is denoted as K. Specifically, when K=1, it corresponds to Algorithm 1 without the inner-consensus loop, which has reduced communication complexity.
- Algorithm 3 is another variant of Algorithm 1, which uses asynchronous parallel updates instead of synchronous parallel updates. The number of asynchronous-parallel-update iterations is denoted as Q. Specifically, when Q=1, it is equivalent to Algorithm 1.
  
<div align="center">
  <img src="https://github.com/cziqin/Automated_Stepsizes/blob/main/figures/mushrooms.png" alt="Fig3" width="900">
</div>

- Fig. a shows the loss evolution of Algorithm 1, Algorithm 2 with K=1, Algorithm 2 with K=10, Algorithm 3 with Q=10, DGM-BB-C with K=1, and DGD over 150 iterations, respectively. The results demonstrate that Algorithm 1 has better convergence accuracy compared with its single-inner-consensus-loop variant (Algorithm 2 with K=1), its asynchronous-parallel-update variant (Algorithm 3 with Q=10), DGM-BB-C [42], and DGD [48].
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
(Get-Content matrix.py) -replace 'CONST_Q = 1', 'CONST_Q = 10' | Set-Content matrix.py
python .\Matrix_factorization\main.py --test_num 0 --iterations 1000
```

4. In this experiment, only the stepsize of DGD needs to be tuned, while Algorithm 1, Algorithm 2, and DGM-BB-C are tuning free. We set the stepsize $\eta=10^{-4}$ for the DGD, since it was the suboptimal stepsize that we found based on the following tuning results after 200 iterations:
   
<table>
  <tr> <th rowspan="2">Algorithms</th> <th colspan="9">Stepsizes</th>
  </tr>
  <tr> <th>$10^{-8}$</th>   <th>$10^{-7}$</th>   <th>$10^{-6}$</th> <th>$10^{-5}$</th>   <th>$10^{-4}$</th>   <th>$10^{-3}$</th>
    <th>$10^{-2}$</th>   <th>$10^{-1}$</th>   <th>$10^{0}$</th>
  </tr>
  <tr>
    <td>DGD</td> <td>5.673 </td>  <td>5.667</td>  <td>5.614</td>  <td>5.330</td>
    <td><b>5.135<b></td>  <td>nan</td>  <td>nan</td>  <td>nan</td>  <td>nan</td>
  </tr>
  <tr>
    <td>Algorithm 1 </td>  <td colspan="9"><b>5.095<b></td>
  </tr>
</table>

>Note: Since the standard deviations of all algorithms are smaller than 0.001, they are omitted in this table.

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
1. You can use the following command to execute Algorithm 1 for the conventional neural network training task on the CIFAR-10 dataset:
```
python .\Neural_networks\main.py --test_num 0 --epochs 70 --batch_size 128 --dataset 'cifar10'
```

>Note: Before running the script, please ensure that the [`CIFAR-10`](https://www.cs.toronto.edu/~kriz/cifar.html) dataset has been downloaded and placed in the `./Neural_networks/data` directory.

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

5. In this experiment, we set the stepsize $\eta=0.005$ for DADAM, $\eta=0.1$ for DAMSGrad, and $\eta=0.5$ for DSGD-N, respectively, since they were the suboptimal stepsizes that we found based on the following tuning results（in terms of test accuracy） after 100 epochs:

<table>
  <tr> 
    <th rowspan="2">Step Sizes</th>    <th colspan="4">Algorithms</th>
  </tr>
  <tr> 
    <th>DADAM</th>   <th>DAMSGrad</th>  <th>DSGD-N</th>   <th>Algorithm 3 (without tuning)</th>
  </tr>
  <tr>
    <td>0.00001</td> <td>0.304 &pm; 0.012</td> <td>0.223 &pm; 0.012</td> <td>0.156 &pm; 0.022</td> <td rowspan="13"><b>0.794 &pm; 0.002</b></td>
  </tr>
  <tr>
    <td>0.00005</td> <td>0.507 &pm; 0.006</td> <td>0.295 &pm; 0.004</td> <td>0.208 &pm; 0.014</td>
  </tr>
  <tr>
    <td>0.0001</td> <td>0.579 &pm; 0.010</td> <td>0.368 &pm; 0.005</td> <td>0.220 &pm; 0.013</td>
  </tr>
  <tr>
    <td>0.0005</td> <td>0.685 &pm; 0.006</td> <td>0.597 &pm; 0.009</td> <td>0.261 &pm; 0.002</td>
  </tr>
  <tr>
    <td>0.001</td> <td>0.709 &pm; 0.009</td> <td>0.654 &pm; 0.019</td> <td>0.323 &pm; 0.002</td>
  </tr>
  <tr>
    <td>0.005</td> <td><b>0.767 &pm; 0.009</b></td> <td>0.713 &pm; 0.018</td> <td>0.571 &pm; 0.007</td>
  </tr>
  <tr>
    <td>0.01</td> <td>0.754 &pm; 0.023</td> <td>0.718 &pm; 0.022</td> <td>0.619 &pm; 0.026</td>
  </tr>
  <tr>
    <td>0.05</td> <td>0.671 &pm; 0.009</td> <td>0.741 &pm; 0.009</td> <td>0.697 &pm; 0.013</td>
  </tr>
  <tr>
    <td>0.1</td> <td>0.646 &pm; 0.026</td> <td><b>0.762 &pm; 0.008</b></td> <td>0.757 &pm; 0.012</td>
  </tr>
  <tr>
    <td>0.5</td> <td>0.634 &pm; 0.027</td> <td>0.668 &pm; 0.008</td> <td><b>0.764 &pm; 0.008</b></td>
  </tr>
  <tr>
    <td>1</td> <td>0.511 &pm; 0.083</td> <td>0.652 &pm; 0.025</td> <td>0.736 &pm; 0.007</td>
  </tr>
  <tr>
    <td>5</td> <td>0.329 &pm; 0.124</td> <td>0.169 &pm; 0.031</td> <td>0.664 &pm; 0.013</td>
  </tr>
  <tr>
    <td>10</td> <td>0.309 &pm; 0.096</td> <td>0.164 &pm; 0.040</td> <td>0.100 &pm; 0.000</td>
  </tr>
</table>


### Experimental results
<div align="center">
  <img src="https://github.com/cziqin/Automated_Stepsizes/blob/main/figures/cifar10.png" alt="Fig5" width="900">
</div>

- Fig. a shows the evolution of the training loss of Algorithm 3 with Q=1, Algorithm 3 with Q=10, DADAM, DAMSGrad, and DSGD-N over epochs. The shaded area represents the 95% confidence interval.
- Fig. b shows the evolution of the test accuracy of Algorithm 3 with Q=1, Algorithm 3 with Q=10, DADAM, DAMSGrad, and DSGD-N over epochs.
- Fig. c shows the measured trace of the average stepsize (of five agents) for the comparision algorithms, with the error bars representing the standard derivation and the solid line representing the nonlinear fitted curve.
- Fig. d shows the comparision of the average stepsize (of five agents) in the five algorithms. Box plots show the median, 1st and 3rd quartiles, and the 5th to 95th percentiles.
- Fig. e shows the comparision results of Algorithm 3 with Q=1 (synchronous parallel updates) with Algorithm 3 under different numbers of asynchronous-parallel-update iterations in terms of communication rounds.
- Fig. f shows the comparision results of Algorithm 3 with ATC-DIGing (with $\eta=0.02$) and DSGD (with $\eta=\frac{0.02}{t+1}$) in terms of communication rounds.

> Note: All experimental results (including training loss, test accuracy, average stepsizes, etc.) will be automatically saved as `.csv` files in the `./Neural_networks/results` directory.

### ImageNet
1. You can use the following command to execute Algorithm 1 for the conventional neural network (CNN) training task on the ImageNet dataset:
```
python .\Neural_networks\main.py --test_num 0 --epochs 20 --batch_size 128 --dataset 'imagenet'
```

>Note: Before running the script, you need to first ensure that the [`ImageNet`](https://academictorrents.com/collection/imagenet-2012) dataset has been downloaded. Next, you should run datadeal.py to split the dataset into training and test sets. Finally, make sure they are placed in the ./Neural_networks/data/imagenet/train and ./Neural_networks/data/imagenet/sort_val directories, respectively. 

2. To execute Algorithm 3 with a desired number of asynchronous-parallel-update iterations $Q$ (e.g., $Q=10$), you need to first set the parameter  `CONST_Q` (e.g., `CONST_Q=10`) in the [`matrix.py`](https://github.com/cziqin/Automated_Stepsizes/blob/main/Neural_networks/matrix.py) file, and then execute Algorithm 3. For example, you can run the following command in the Windows PowerShell:
```
(Get-Content matrix.py) -replace 'CONST_Q = 1', 'CONST_Q = 10' | Set-Content matrix.py
python .\Neural_networks\main.py --test_num 0 --epochs 70 --batch_size 128 --dataset 'imagenet'
```

3. To specify the print interval (e.g., printing the training loss, test accuracy, and average stepsize every 10 iterations), you need to first update the parameter `SPECIFIC_LOG_INTERVAL` (e.g., `SPECIFIC_LOG_INTERVAL=10`) in the [`matrix.py`](https://github.com/cziqin/Automated_Stepsizes/blob/main/Neural_networks/matrix.py) file, and then execute Algorithm 3. 

4. To specify the random seed used in training (e.g., setting seed=42), you can first update the parameter `SEED` (e.g., `SEED = 42`) in the [`matrix.py`](https://github.com/cziqin/Automated_Stepsizes/blob/main/Neural_networks/matrix.py) file, and then execute Algorithm 3. The default seed is 42.

5. In this experiment, we set the same step size for DADAM, DAMSGrad, and DSGD-N as those used in the CIFAR-10 experiment, because tuning stepsizes for them in CNN training on the large-scale ImageNet dataset (which consists of over 1.28 million images) would cost a substantial amount of time. 

## 🚀 Discussions
In this repository, we have provided the codes to implement our automated stepsize approach in a fully distributed manner using three machine learning tasks on benchmark datasets. To the best of our knowledge, it is the first to completely avoid manual stepsize adjustment in decentralized machine learning tasks without the assistant of any centralized aggregators or information of the global objective function. Compared with existing counterpart adaptive/nonadpative algorithms, our approach achives a faster convergence speed and a higher convergence/learning accuracy. 

## License
Distributed under the MIT License. See [`LICENSE`](https://github.com/cziqin/Automated_Stepsizes/blob/main/LICENSE) for more information.
## Authors
- [Ziqin Chen](https://scholar.google.com/citations?user=i-IM2rIAAAAJ&hl=zh-CN)
- [Yongqiang Wang](https://scholar.google.com/citations?hl=zh-CN&user=shSZpGUAAAAJ)
