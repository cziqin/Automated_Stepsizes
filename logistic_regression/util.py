import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as la

import csv
from sklearn.utils import shuffle
from sklearn.datasets import load_svmlight_file
from matrix import generate_Ei_matrix, distributively_estimate_consensus_params
import math

from loss_funcs import logistic_loss, logistic_gradient


def load_data(stratified, data_path, agents=10):
    agent_samples = {}
    agent_targets = {}
    agent_L = {}
    agent_L2 = {}

    data = load_svmlight_file(data_path)
    total_samples, total_targets = data[0].toarray(), data[1]
    if (np.unique(total_targets) == [1, 2]).all():
        total_targets -= 1

    total_samples, total_targets = shuffle(total_samples, total_targets)
    leng = int(len(total_samples))
    agent_sample_len, remainder = int(
        leng // agents), leng % agents

    if stratified:
        print('Hello')
        for i in range(agents):
            agent_samples[i] = total_samples[
                               i * agent_sample_len:i * agent_sample_len + agent_sample_len]
            agent_targets[i] = total_targets[i * agent_sample_len:i * agent_sample_len + agent_sample_len]
            agent_L[i] = logistic_smoothness(agent_samples[i])
            agent_L2[i] = agent_L[i] / (20 * agent_samples[i].shape[0])

    else:
        print('Yes')
        targets_0 = np.array(list(map(lambda x: x, np.where(total_targets == 0))))[
            0]
        targets_1 = np.array(list(map(lambda x: x, np.where(total_targets == 1))))[
            0]
        print(int(len(targets_0)))
        print(int(len(targets_1)))

        agent_samples[0] = total_samples[targets_0[:agent_sample_len]]
        agent_targets[0] = total_targets[targets_0[:agent_sample_len]]
        agent_samples[1] = total_samples[targets_0[agent_sample_len * 1:agent_sample_len * 2]]
        agent_targets[1] = total_targets[targets_0[agent_sample_len * 1:agent_sample_len * 2]]
        agent_samples[2] = total_samples[targets_0[agent_sample_len * 2:agent_sample_len * 3]]
        agent_targets[2] = total_targets[targets_0[agent_sample_len * 2:agent_sample_len * 3]]
        agent_samples[3] = total_samples[targets_1[:agent_sample_len]]
        agent_targets[3] = total_targets[targets_1[:agent_sample_len]]
        agent_samples[4] = total_samples[targets_1[agent_sample_len * 1:agent_sample_len * 2]]
        agent_targets[4] = total_targets[targets_1[agent_sample_len * 1:agent_sample_len * 2]]
        agent_samples[5] = total_samples[targets_1[agent_sample_len * 2:agent_sample_len * 3]]
        agent_targets[5] = total_targets[targets_1[agent_sample_len * 2:agent_sample_len * 3]]
        targets_0 = np.delete(targets_0, [range(0, agent_sample_len * 3)])
        targets_1 = np.delete(targets_1, [range(0, agent_sample_len * 3)])
        remaining_samples = total_samples[np.append(targets_0, targets_1)]
        remaining_targets = total_targets[np.append(targets_0, targets_1)]
        total_remain_samples, total_remain_targets = shuffle(remaining_samples, remaining_targets)

        agent_samples[6] = remaining_samples[:agent_sample_len]
        agent_targets[6] = remaining_targets[:agent_sample_len]
        agent_samples[7] = remaining_samples[agent_sample_len * 1:agent_sample_len * 2]
        agent_targets[7] = remaining_targets[agent_sample_len * 1:agent_sample_len * 2]
        agent_samples[8] = remaining_samples[agent_sample_len * 2:agent_sample_len * 3]
        agent_targets[8] = remaining_targets[agent_sample_len * 2:agent_sample_len * 3]
        agent_samples[9] = remaining_samples[agent_sample_len * 3:agent_sample_len * 4]
        agent_targets[9] = remaining_targets[agent_sample_len * 3:agent_sample_len * 4]
        for i in range(agents):
            agent_L[i] = logistic_smoothness(agent_samples[i])
            agent_L2[i] = agent_L[i] / (20 * agent_samples[i].shape[0])

    return agent_samples, agent_targets, agent_L, agent_L2


def logistic_smoothness(samples, covtype=True):
    if covtype:
        return 0.25 * np.max(la.eigvalsh(samples.T @ samples / (samples.shape[0])))
    else:
        return 0.25 * np.max(la.eigvalsh(samples.T @ samples / (samples.shape[0])))


class Trainer:
    def __init__(self, data, agent_matrix, iterations=1001,
                 min_allow=0):
        self.momentum_param = 0.3
        self.momentum = {}
        self.t = None
        self.iterations_tot = iterations
        self.min_allow = min_allow
        self.agent_matrix = agent_matrix
        self.agents = agent_matrix.shape[0]
        self.grad_norm = []
        self.eta = {}
        self.eta_record = {}
        self.name = ""
        self.final_losses_plotted = []
        self.final_grad_plotted = []
        self.losses = []

        self.agent_samples = data[0]
        self.agent_targets = data[1]
        self.agent_L = data[2]
        self.agent_L2 = data[3]
        self.agent_parameters = {}
        self.vector = np.zeros(self.agent_samples[0].shape[1])
        self.vector_1 = np.ones_like(self.agent_samples[0].shape[1])
        for i in range(self.agents):
            self.agent_parameters[i] = [np.zeros(self.agent_samples[i].shape[1])]

    def compute_loss(self, iteration):
        temp_loss = 0
        for i in range(self.agents):
            temp_loss += logistic_loss(self.agent_parameters[i][iteration], self.agent_samples[i],
                                       self.agent_targets[i], self.agent_L2[i])
        if iteration == 0:
            self.losses = [temp_loss / self.agents]
        else:
            self.losses.append(temp_loss / self.agents)

    def compute_grad(self, agent_num, iterations, init=False, save=True):
        params = self.agent_parameters[agent_num][iterations]

        samples = self.agent_samples[agent_num]
        targets = self.agent_targets[agent_num]
        l2 = self.agent_L2[agent_num]
        if iterations == 0:
            self.grads[agent_num] = [logistic_gradient(params, samples, targets, l2)]
            return self.grads[agent_num]
        else:
            self.grads[agent_num].append(logistic_gradient(params, samples, targets, l2))
        return la.norm(self.grads[agent_num][-1]) > self.min_allow

    def train(self):
        self.losses = []
        self.iterations = []
        self.grads = {}
        self.et = []
        for j in range(self.agents):
            self.compute_grad(agent_num=j, iterations=0)
        self.init_run()
        self.compute_loss(iteration=0)
        for j in range(self.agents):
            self.grad_norm = [np.linalg.norm(sum([grad[-1] for grad in self.grads.values()]) / self.agents)]
        self.iterations = [0]

        for t in range(self.iterations_tot):
            self.compute_loss(iteration=t)
            for j in range(self.agents):
                self.compute_grad(agent_num=j, iterations=t)
            self.grad_norm.append(np.linalg.norm(sum([grad[-1] for grad in self.grads.values()]) / self.agents))
            self.iterations.append(t)
            self.step(t)

            self.et.append(sum([et[-1] for et in self.eta.values()]) / self.agents)

            if t % 10 == 0:
                print(
                    f"Optimizer: {self.name}, Iteration: {self.iterations[t]}, Gradients: {self.grad_norm[t]}, Eta: {self.et[t]}")

    def save_data(self, filename, skip, iterations):
        with open(filename, mode="a") as csv_file:
            file = csv.writer(csv_file, lineterminator="\n")
            file.writerow([self.name])
            file.writerow(self.iterations[::skip])
            file.writerow(self.grad_norm[::skip])
            file.writerow(self.et[::skip])
            file.writerow([])


def plot_all_losses(optimizers, skip):
    markerstyles = ["d", "*", "s", "p", "8", "h", "x", "o"]
    f_opt = np.min([np.min(opt.losses) for opt in optimizers])
    for i, opt in enumerate(optimizers):
        opt.plot_loss_value(f_opt, markerstyles[i], skip=skip)
    plt.ylabel("F(x)-F*")
    plt.xlabel("Iteration")
    plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    plt.grid(visible=True)
    plt.savefig("LogicLoss.jpg")
    plt.clf()


def plot_grad_norm(optimizer, skip):
    markerstyles = ["d", "*", "s", "p", "8", "h", "x", "o"]
    for i, opt in enumerate(optimizer):
        opt.plot_grad_norm(markerstyles[i], skip=skip)
    plt.ylabel(r"$\left\| \nabla F(x) \right\|$")
    plt.xlabel("Iteration")
    plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    plt.grid(visible=True)
    plt.savefig("LogicNorm.jpg")
    plt.clf()
