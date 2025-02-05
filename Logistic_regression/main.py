import argparse
import os
from optimizers import *
from matrix import *
from train import load_data
import numpy as np
import swanlab

np.random.seed(42)

agents = 5
dataset = "mushrooms"

agent_matrix = cycle_graph(agents)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--test_num", type=int, default=1)
    parser.add_argument("-s", "--stratified", default=False)
    parser.add_argument("-i", "--iterations", type=int, default=16100)

    return parser.parse_args()

args = parse_args()
iterations = args.iterations
stratified = args.stratified

cwd = os.getcwd()
data_path = os.path.join(cwd, "mushrooms")
data = load_data(data_path=data_path, agents=agents, stratified=stratified)

Algorithm_name = None
if args.test_num == 0:
    Algorithm_name = "Algorithm1"
elif args.test_num == 1:
    Algorithm_name = "Algorithm3"
elif args.test_num == 2:
    Algorithm_name = "Algorithm4"
elif args.test_num == 3:
    Algorithm_name = "DBBG"
elif args.test_num == 4:
    Algorithm_name = "DGD"

swanlab.init(
    experiment_name=f"{Algorithm_name}-{dataset}",
    logdir="./logs",
    mode='local',
)

optimizers = None
if args.test_num == 0:
    algorithm1 = Algorithm1(agent_matrix=agent_matrix, iterations=iterations, data=data)
    algorithm1.train()
    optimizers = [algorithm1]
elif args.test_num == 1:
    algorithm3 = Algorithm3(agent_matrix=agent_matrix, iterations=iterations, data=data)
    algorithm3.train()
    optimizers = [algorithm3]
elif args.test_num == 2:
    algorithm4 = Algorithm4(agent_matrix=agent_matrix, iterations=iterations, data=data)
    algorithm4.train()
    optimizers = [algorithm4]
elif args.test_num == 3:
    dbbg = DBBG(agent_matrix=agent_matrix, iterations=iterations, data=data)
    dbbg.train()
    optimizers = [dbbg]
elif args.test_num == 4:
    dgd = DGD(agent_matrix=agent_matrix, iterations=iterations, data=data)
    dgd.train()
    optimizers = [dgd]


results_path = os.path.join(cwd, "results")
if not os.path.isdir(results_path):
    os.mkdir(results_path)

filename = os.path.join(results_path, f"{Algorithm_name}.csv")
for opt in optimizers:
    opt.save_data(filename, skip=40)
