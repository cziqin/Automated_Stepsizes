import numpy as np

from train import Trainer
import csv
import os
from loss_function import *
from matrix import finite_consensus, imperfect_consensus,Subrounte1

class Algorithm1(Trainer):
    def __init__(self, *args, **kwargs):
        super(Algorithm1, self).__init__(*args, **kwargs)
        self.name = "Algorithm1"
        self.agent_y = {}
        self.gamma=Subrounte1(self.agent_matrix,self.agents)

    def step(self, iteration):
        x = np.zeros((self.agents, self.agent_parameters[0][iteration].shape[0]))
        y = np.zeros((self.agents, self.grads[0][iteration].shape[0]))

        if iteration == 0:
            self.agent_y = {i: [self.grads[i][iteration]] for i in range(self.agents)}

        # parameter x update
        for i in range(self.agents):
            y[i] = self.agent_y[i][iteration]
            x[i] = self.agent_parameters[i][iteration]
            x[i] += - self.eta[i][iteration] * y[i]

        # fixed-time consensus
        Phi = finite_consensus(x, self.agent_matrix, self.agents,self.gamma)

        # parameter y update
        for i in range(self.agents):
            samples = self.agent_samples[i]
            targets = self.agent_targets[i]
            l2 = self.agent_L2[i]
            new_grads = logistic_gradient(Phi[i], samples, targets, l2)
            y[i]+=new_grads-self.grads[i][iteration]

        Theta = finite_consensus(y, self.agent_matrix, self.agents, self.gamma)

        # automated_stepsize_update
        for i in range(self.agents):
            self.agent_parameters[i].append(Phi[i])
            self.agent_y[i].append(Theta[i])

            if iteration == 0:
                self.eta[i].append(self.eta[i][0])
            else:
                a = np.sqrt(1 + self.eta[i][iteration] / self.eta[i][iteration - 1]) * self.eta[i][
                    iteration]
                b = la.norm(self.agent_parameters[i][iteration + 1] - self.agent_parameters[i][iteration]) / (
                        la.norm(self.agent_y[i][iteration + 1] - self.agent_y[i][iteration]))
                self.eta[i].append(min(a, b))

    def init_run(self):
        for i in range(self.agents):
            self.eta[i] = [1 / self.agent_L[i]]


class Algorithm3(Trainer):
    def __init__(self, *args, **kwargs):
        super(Algorithm3, self).__init__(*args, **kwargs)
        self.name = "Algorithm3"
        self.agent_y = {}
        self.Q = 80
        self.gamma = Subrounte1(self.agent_matrix, self.agents)

    def step(self, iteration):
        x = np.zeros((self.agents, self.agent_parameters[0][iteration].shape[0]))
        y = np.zeros((self.agents, self.grads[0][iteration].shape[0]))

        if iteration == 0:
            self.agent_y = {i: [self.grads[i][iteration]] for i in range(self.agents)}

        # parameter x update
        for i in range(self.agents):
            y[i] = self.agent_y[i][iteration]
            x[i] = self.agent_parameters[i][iteration]
            x[i] += - self.eta[i][iteration] * y[i]

        # fixed-time consensus
        if iteration % self.Q == 0:
            x= finite_consensus(x, self.agent_matrix, self.agents,self.gamma)

        # parameter y update
        for i in range(self.agents):
            samples = self.agent_samples[i]
            targets = self.agent_targets[i]
            l2 = self.agent_L2[i]
            new_grads = logistic_gradient(x[i], samples, targets, l2)
            y[i]+=new_grads-self.grads[i][iteration]

        if iteration % self.Q == 0:
            y = finite_consensus(y, self.agent_matrix, self.agents, self.gamma)

        # automated_stepsize_update
        for i in range(self.agents):
            self.agent_parameters[i].append(x[i])
            self.agent_y[i].append(y[i])

            if iteration == 0:
                self.eta[i].append(self.eta[i][0])
            else:
                a = np.sqrt(1 + self.eta[i][iteration] / self.eta[i][iteration - 1]) * self.eta[i][
                    iteration]
                b = la.norm(self.agent_parameters[i][iteration + 1] - self.agent_parameters[i][iteration]) / (
                        la.norm(self.agent_y[i][iteration + 1] - self.agent_y[i][iteration]))
                self.eta[i].append(min(a, b))

    def init_run(self):
        for i in range(self.agents):
            self.eta[i] = [1 / self.agent_L[i]]

class Algorithm4(Trainer):
    def __init__(self, *args, **kwargs):
        super(Algorithm4, self).__init__(*args, **kwargs)
        self.name = "Algorithm4"
        self.K = 10 # Change this to 1 for Algorithm S1
        self.agent_y = {}

    def step(self, iteration):
        x = np.zeros((self.agents, self.agent_parameters[0][iteration].shape[0]))
        y = np.zeros((self.agents, self.grads[0][iteration].shape[0]))

        if iteration == 0:
            self.agent_y = {i: [self.grads[i][iteration]] for i in range(self.agents)}

        # parameter update
        for i in range(self.agents):
            y[i] = self.agent_y[i][iteration]
            x[i] = self.agent_parameters[i][iteration]
            x[i] += - self.eta[i][iteration] * y[i]

        # imperfect-consensus_x
        x = imperfect_consensus(x, self.K, self.agent_matrix,self.agents)

        for i in range(self.agents):
            samples = self.agent_samples[i]
            targets = self.agent_targets[i]
            l2 = self.agent_L2[i]
            new_grads = logistic_gradient(x[i], samples, targets, l2)
            y[i] += new_grads - self.grads[i][iteration]

        # imperfect-consensus_y
        y = imperfect_consensus(y, self.K, self.agent_matrix,self.agents)

        for i in range(self.agents):
            self.agent_parameters[i].append(x[i])
            self.agent_y[i].append(y[i])
            if iteration == 0:
                self.eta[i].append(self.eta[i][0])
            else:
                a = np.sqrt(1 + self.eta[i][iteration] / self.eta[i][iteration - 1]) * self.eta[i][
                    iteration]
                b = la.norm(self.agent_parameters[i][iteration + 1] - self.agent_parameters[i][iteration]) / (
                        la.norm(self.agent_y[i][iteration + 1] - self.agent_y[i][iteration]))
                self.eta[i].append(min(a, b))

    def init_run(self):
        for i in range(self.agents):
            self.eta[i] = [1 / self.agent_L[i]]


class DBBG(Trainer):
    def __init__(self, *args, **kwargs):
        super(DBBG, self).__init__(*args, **kwargs)
        self.name = "DBBG"
        self.K = 10
        self.agent_y = {}

    def step(self, iteration):
        x = np.zeros((self.agents, self.agent_parameters[0][iteration].shape[0]))
        y = np.zeros((self.agents, self.grads[0][iteration].shape[0]))
        old_grads = {}

        # update stepsizes
        for i in range(self.agents):
            if iteration == 0:
                    pass
            else:
                a = la.norm(np.dot((self.agent_parameters[i][iteration] - self.agent_parameters[i][iteration - 1]).T,
                                   (self.agent_parameters[i][iteration] - self.agent_parameters[i][iteration - 1]))) \
                    / la.norm(np.dot((self.agent_parameters[i][iteration] - self.agent_parameters[i][iteration - 1]).T,
                                     (self.grads[i][iteration] - self.grads[i][iteration - 1])))
                b = la.norm(np.dot((self.agent_parameters[i][iteration] - self.agent_parameters[i][iteration - 1]).T,
                                   (self.grads[i][iteration] - self.grads[i][iteration - 1]))) \
                    / la.norm(np.dot((self.grads[i][iteration] - self.grads[i][iteration - 1]).T,
                                     (self.grads[i][iteration] - self.grads[i][iteration - 1])))
                # self.eta[i].append(min(a, b))
                self.eta[i].append(min(a, b, (100*self.average_L)/self.agents))
        # parameter update
        if iteration == 0:
            self.agent_y = {i: [self.grads[i][iteration]] for i in range(self.agents)}
        for i in range(self.agents):
            y[i] = self.agent_y[i][iteration]
            x[i] = self.agent_parameters[i][iteration]

            samples = self.agent_samples[i]
            targets = self.agent_targets[i]
            l2 = self.agent_L2[i]

            old_grads[i] = logistic_gradient(x[i], samples, targets, l2)
            x[i] = x[i] - self.eta[i][iteration] * y[i]

        # imperfect-consensus_x
        x = imperfect_consensus(x, self.K, self.agent_matrix,self.agents)

        for i in range(self.agents):
            samples = self.agent_samples[i]
            targets = self.agent_targets[i]
            l2 = self.agent_L2[i]
            new_grads = logistic_gradient(x[i], samples, targets, l2)
            y[i] = y[i] + new_grads - old_grads[i]

        # imperfect-consensus_y
        y = imperfect_consensus(y, self.K, self.agent_matrix,self.agents)

        for i in range(self.agents):
            self.agent_parameters[i].append(x[i])
            self.agent_y[i].append(y[i])

    def init_run(self):
        self.average_L = 0
        for i in range(self.agents):
            self.eta[i] = [2 / self.agent_L[i]]
            self.average_L += 1 / self.agent_L[i]



class DGD(Trainer):
    def __init__(self, *args, **kwargs):
        super(DGD, self).__init__(*args, **kwargs)
        self.name = "DGD"
        self.K = 1

    def step(self, iteration):
        x = np.zeros((self.agents, self.agent_parameters[0][iteration].shape[0]))
        for i in range(self.agents):
            x[i] = self.agent_parameters[i][iteration]

        # imperfect-consensus_x
        x = imperfect_consensus(x, self.K, self.agent_matrix, self.agents)
        for i in range(self.agents):
            self.agent_parameters[i].append(x[i] - self.eta[i][iteration] * self.grads[i][iteration])
            self.eta[i].append(1 / self.agent_L[i])

    def init_run(self):
        for i in range(self.agents):
            self.eta[i] = [1 / self.agent_L[i]]


