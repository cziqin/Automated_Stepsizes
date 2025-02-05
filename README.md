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
2. Download and install [Anaoconda](https://www.anaconda.com) (if you don't have it already)
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
├── logistic_regression                 # Directory to implement your custom algorithm/trainable/agent
│   ├── custom_random_agent
│   ├── random_policy
│   ├── __init__.py
│   └── registry.py                     # Register your custom agents here
├── envs
│   ├── __init__.py
│   ├── framestack.py                   # Example for using custom env wrappers
│   ├── procgen_env_wrapper.py          # Base env used during evaluations (DO NOT EDIT)
├── experiments                         # Directory contaning the config for different experiments
│   ├── impala-baseline.yaml            # Baseline using impala
│   ├── procgen-starter-example.yaml    # Sample experiment config file
│   └── random-policy.yaml              # Sample random policy config file
├── models                              # Directory to implement custom models
│   ├── impala_cnn_tf.py
│   ├── impala_cnn_torch.py
│   └── my_vision_network.py
├── preprocessors                       # Directory to implement your custom observation wrappers
│   ├── __init__.py                     # Register your preprocessors here
│   └── custom_preprocessor.py
├── utils                               # Helper scripts for the competition
│   ├── setup.sh                        # Setup local procgen environment using `conda`
│   ├── submit.sh                       # Submit your solution
│   ├── teardown.sh                     # Remove the existing local procgen environment using `conda`
│   ├── validate_config.py              # Validate the experiment YAML file
│   └── loader.py
├── Dockerfile                          # Docker config for your submission environment
├── aicrowd.json                        # Submission config file (required)
├── callbacks.py                        # Custom Callbacks & Custom Metrics
├── requirements.txt                    # These python packages will be installed using `pip`
├── rollout.py                          # Rollout script (DO NOT EDIT)
├── run.sh                              # Entrypoint to your submission
└── train.py                            # Script to trigger the training using `rllib` (DO NOT EDIT)

```

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
