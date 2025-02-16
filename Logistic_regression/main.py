import argparse
import os
from optimizers import *
from matrix import *
from train import load_data
import numpy as np

np.random.seed(42)

agents = 5
dataset = "mushrooms"

agent_matrix = cycle_graph(agents)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--test_num", type=int, default=0)
    parser.add_argument("-s", "--stratified", default=False)
    parser.add_argument("-i", "--iterations", type=int, default=1000)

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
    Algorithm_name = "Algorithm2"
elif args.test_num == 2:
    Algorithm_name = "DBBC"
elif args.test_num == 3:
    Algorithm_name = "DGD"

optimizers = None
if args.test_num == 0:
    algorithm1 = Algorithm1(agent_matrix=agent_matrix, iterations=iterations, data=data)
    algorithm1.train()
    optimizers = [algorithm1]
elif args.test_num == 1:
    algorithm2 = Algorithm2(agent_matrix=agent_matrix, iterations=iterations, data=data)
    algorithm2.train()
    optimizers = [algorithm2]
elif args.test_num == 2:
    dbbc = DBBC(agent_matrix=agent_matrix, iterations=iterations, data=data)
    dbbc.train()
    optimizers = [dbbc]
elif args.test_num == 3:
    dgd = DGD(agent_matrix=agent_matrix, iterations=iterations, data=data)
    dgd.train()
    optimizers = [dgd]


results_path = os.path.join(cwd, "results")
if not os.path.isdir(results_path):
    os.mkdir(results_path)

filename = os.path.join(results_path, f"{Algorithm_name}.csv")
for opt in optimizers:
    opt.save_data(filename, skip=10)
