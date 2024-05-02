import argparse
import os
import random
import numpy as np
from matrix import cycle_graph, fully_connected_graph

from train import *
import swanlab

agents = 10
w = cycle_graph(agents)

# dataset = "mnist"
dataset = "mnist"
epochs = 6
if dataset == "mnist":
    bs = 40
else:
    bs = 128
seed = 42


def parse_args():
    """ Function parses command line arguments """
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--test_num", default=0, type=int)
    parser.add_argument("-r", "--run_num", default=0, type=int)
    parser.add_argument("-s", "--stratified", default=True)
    return parser.parse_args()


args = parse_args()
cwd = os.getcwd()
results_path = os.path.join(cwd, "results")
if not os.path.isdir(results_path):
    os.mkdir(results_path)

stratified = args.stratified
fname = os.path.join(results_path, f"{dataset}_e{epochs}_hom{stratified}_{args.test_num}.csv")

print(f"Test Num {args.test_num}, run num: {args.run_num}, {fname}")

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

if args.test_num == 0:
    Algorithm2Trainer(dataset=dataset, batch_size=bs, epochs=epochs, w=w, fname=fname, stratified=stratified)
elif args.test_num == 1:
    AlgorithmS1Trainer(dataset=dataset, batch_size=bs, epochs=epochs, w=w, fname=fname, stratified=stratified)
elif args.test_num == 2:
    DSGDNTrainer(dataset=dataset, batch_size=bs, epochs=epochs, w=w, fname=fname, stratified=stratified)
elif args.test_num == 3:
    DADAMTrainer(dataset=dataset, batch_size=bs, epochs=epochs, w=w, fname=fname, stratified=stratified)
elif args.test_num == 4:
    DAMSGradTrainer(dataset=dataset, batch_size=bs, epochs=epochs, w=w, fname=fname, stratified=stratified)
