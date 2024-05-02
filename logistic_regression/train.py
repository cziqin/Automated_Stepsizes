import argparse
import os
import matplotlib.pyplot as plt

from optimizers import *
from matrix import cycle_graph, fully_connected_graph
from util import load_data, plot_all_losses, plot_grad_norm

agents = 10
agent_matrix = cycle_graph(agents)

dataset = "mushrooms"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--test_num", default=0, type=int)
    parser.add_argument("-s", "--stratified", default=True)
    parser.add_argument("-k", "--kappa", type=float, default=0.4)
    parser.add_argument("-i", "--iterations", type=int, default=1010)

    return parser.parse_args()


args = parse_args()
cwd = os.getcwd()
data_path = os.path.join(cwd, "mushrooms")
stratified = args.stratified
label = "homogeneous" if stratified else "heterogeneous"

data = load_data(stratified=stratified, data_path=data_path, agents=agents)

loop_num = 2
fix_theta = 1.6
iterations = args.iterations
decay = 0.3
kap = args.kappa


if args.test_num == 0:
    algorithm1 = Algorithm1(agent_matrix=agent_matrix, iterations=iterations, data=data)
    algorithm1.train()
    optimizers = [algorithm1]
elif args.test_num == 1:
    algorithm2 = Algorithm2(agent_matrix=agent_matrix, iterations=iterations, data=data)
    algorithm2.train()
    optimizers = [algorithm2]
elif args.test_num == 2:
    dbbg = DBBG(agent_matrix=agent_matrix, iterations=iterations, data=data)
    dbbg.train()
    optimizers = [dbbg]
elif args.test_num == 3:
    dgd = DGD(agent_matrix=agent_matrix, iterations=iterations, data=data)
    dgd.train()
    optimizers = [dgd]


results_path = os.path.join(cwd, "results")
if not os.path.isdir(results_path):
    os.mkdir(results_path)

filename = os.path.join(results_path, f"{dataset}_{label}.csv")
for opt in optimizers:
    opt.save_data(filename, skip=10, iterations=iterations)
