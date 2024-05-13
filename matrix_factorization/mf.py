import argparse
import os

import pandas as pd
from matrix import cycle_graph, fully_connected_graph
from mfutil import *


agents = 10
agent_matrix = fully_connected_graph(agents, 0.05)
skip = 10

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--test_num", default=0, type=int)
    parser.add_argument("-k", "--kappa", default=0.9, type=float)
    parser.add_argument("-s", "--stratified", default=1, type=int)
    parser.add_argument("-i", "--iterations", type=int, default=1010)
    return parser.parse_args()


def load_data(stratified, agent_num=10):
    agent_A = {}
    names = ['user_id', 'item_id', 'rating', 'timestamp']
    df = pd.read_csv('./u.data', sep='\t', names=names)
    n_users = df.user_id.unique().shape[0]
    n_items = df.item_id.unique().shape[0]
    ratings = np.zeros((n_users, n_items))
    for n in range(agent_num):
        agent_A[n] = ratings.copy()
    if stratified:
        for row in df.itertuples():
            ratings[row[1] - 1, row[2] - 1] = row[3]
        for n in range(agent_num):
            agent_A[n] = ratings
    else:
        for row in df.itertuples():
            agent_A[int(row[3] - 1)][row[1] - 1, row[2] - 1] = row[3]
    return agent_A


args = parse_args()
cwd = os.getcwd()
np.random.seed(0)

start = True if args.stratified == 1 else False
type_t = "homogeneous" if start else "heterogeneous"
A = load_data(start, agents)
iterations = args.iterations

L = 1e5
n_L = 1e5

kap = args.kappa


if args.test_num == 0:
    algorithm1 = Algorithm1(agent_matrix, A, cgrad, L=L, agents=agents, iterations=iterations, skip=skip)
    optimizers = [algorithm1]
elif args.test_num == 1:
    algorithms1 = Algorithms1(agent_matrix, A, cgrad, L=L, agents=agents, iterations=iterations, skip=skip)
    optimizers = [algorithms1]
elif args.test_num == 2:
    dbbg = DBBG(agent_matrix, A, cgrad, L=L, agents=agents, iterations=iterations, skip=skip)
    optimizers = [dbbg]
elif args.test_num == 3:
    cdgdp = CDGDP(agent_matrix, A, cgrad, L=L, agents=agents, iterations=iterations, skip=skip)
    optimizers = [cdgdp]

results_path = os.path.join(cwd, "results")
if not os.path.isdir(results_path):
    os.mkdir(results_path)

filename = os.path.join(results_path, f"{type_t}_it{iterations}.csv")
record_results(filename, optimizers, skip=skip)
