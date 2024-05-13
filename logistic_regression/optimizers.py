import numpy as np

import numpy.linalg as la

from util import Trainer


class Algorithm1(Trainer):
    def __init__(self, *args, **kwargs):
        super(Algorithm1, self).__init__(*args, **kwargs)
        self.name = "Algorithm1"
        self.agent_y = {}
        self.eta = {}

    def step(self, iteration):
        x = {}
        y = {}
        for i in range(self.agents):
            x[i] = [np.zeros(self.agent_samples[i].shape[1])]
            y[i] = [np.zeros(self.agent_samples[i].shape[1])]
        if iteration == 0:
            summation_x = np.zeros(self.agent_samples[0].shape[1])
            for i in range(self.agents):
                self.agent_y[i] = [self.grads[i][iteration]]
                x[i][0] = self.agent_parameters[i][iteration] - self.eta[i][iteration] * self.agent_y[i][iteration]
                summation_x += x[i][0]
            for i in range(self.agents):
                self.agent_parameters[i].append((1 / self.agents) * summation_x)

        else:
            summation_y = np.zeros(self.agent_samples[0].shape[1])

            for i in range(self.agents):
                y[i][0] = self.agent_y[i][iteration - 1] + self.grads[i][iteration] - self.grads[i][iteration - 1]
                summation_y += y[i][0]

            for i in range(self.agents):
                self.agent_y[i].append((1 / self.agents) * summation_y)

            for i in range(self.agents):
                if iteration == 1:
                    self.eta[i].append(self.eta[i][0])
                else:
                    a = np.sqrt(1 + self.eta[i][iteration - 1] / self.eta[i][iteration - 2]) * self.eta[i][
                        iteration - 1]
                    b = la.norm(self.agent_parameters[i][iteration] - self.agent_parameters[i][iteration - 1]) / (
                        la.norm(self.agent_y[i][iteration] - self.agent_y[i][iteration - 1]))
                    self.eta[i].append(min(a, b))

            summation_x = np.zeros(self.agent_samples[0].shape[1])
            for i in range(self.agents):
                x[i][0] = self.agent_parameters[i][iteration] - self.eta[i][iteration] * self.agent_y[i][iteration]
                summation_x += x[i][0]
            for i in range(self.agents):
                self.agent_parameters[i].append((1 / self.agents) * summation_x)

    def init_run(self):
        for i in range(self.agents):
            self.eta[i] = [1 / self.agent_L[i]]


class Algorithm2(Trainer):
    def __init__(self, *args, **kwargs):
        super(Algorithm2, self).__init__(*args, **kwargs)
        self.name = "Algorithm2"
        self.loop_num = 2
        self.fix_theta = np.sqrt(2)
        self.agent_y = {}
        self.eta = {}

    def step(self, iteration):
        x = {}
        y = {}
        for i in range(self.agents):
            x[i] = [np.zeros(self.agent_samples[i].shape[1])]
            y[i] = [np.zeros(self.agent_samples[i].shape[1])]
        if iteration == 0:
            for i in range(self.agents):
                self.agent_y[i] = [self.grads[i][iteration]]
                x[i][0] = self.agent_parameters[i][iteration] - self.eta[i][iteration] * self.agent_y[i][iteration]
            for i in range(self.agents):
                summation_x = np.zeros_like(self.agent_parameters[i][iteration])
                for j in range(self.agents):
                    summation_x += self.agent_matrix[i, j] * x[j][0]
                self.agent_parameters[i].append(summation_x)


        else:
            for i in range(self.agents):
                y[i][0] = self.agent_y[i][iteration - 1] + self.grads[i][iteration] - self.grads[i][iteration - 1]
            for i in range(self.agents):
                summation_y = np.zeros_like(self.agent_parameters[i][iteration])
                for j in range(self.agents):
                    summation_y += self.agent_matrix[i, j] * y[j][0]
                self.agent_y[i].append(summation_y)

            for i in range(self.agents):
                a = self.fix_theta * self.eta[i][iteration - 1]
                b = la.norm(self.agent_parameters[i][iteration] - self.agent_parameters[i][iteration - 1]) / (
                        2.0 * la.norm(self.agent_y[i][iteration] - self.agent_y[i][iteration - 1]))
                c_2 = 1e+10
                c_1 = 1e-10
                self.eta[i].append(max(min(a, b, c_2), c_1))

            for i in range(self.agents):
                x[i][0] = self.agent_parameters[i][iteration] - self.eta[i][iteration] * self.agent_y[i][iteration]
            for i in range(self.agents):
                summation_x = np.zeros_like(self.agent_parameters[i][iteration])
                for j in range(self.agents):
                    summation_x += self.agent_matrix[i, j] * x[j][0]
                self.agent_parameters[i].append(summation_x)

    def init_run(self):
        for i in range(self.agents):
            self.eta[i] = [1 / self.agent_L[i]]


class DBBG(Trainer):
    def __init__(self, *args, **kwargs):
        super(DBBG, self).__init__(*args, **kwargs)
        self.name = "DBBG"
        self.loop_num = 2
        self.agent_y = {}
        self.eta = {}
        self.t = {}

    def step(self, iteration):
        x = {}
        y = {}
        summation_x = {}
        summation_y = {}
        for i in range(self.agents):
            x[i] = [np.zeros(self.agent_samples[i].shape[1])]
            y[i] = [np.zeros(self.agent_samples[i].shape[1])]
        if iteration == 0:
            for i in range(self.agents):
                self.agent_y[i] = [self.grads[i][iteration]]
                x[i][0] = self.agent_parameters[i][iteration] - self.eta[i][iteration] * self.agent_y[i][iteration]
            for m in range(self.loop_num):
                for i in range(self.agents):
                    summation_x[i] = np.zeros_like(self.agent_parameters[i][iteration])
                    for j in range(self.agents):
                        summation_x[i] += self.agent_matrix[i, j] * x[j][m]
                    x[i].append(summation_x[i])
            for i in range(self.agents):
                self.agent_parameters[i].append(x[i][self.loop_num])

        else:
            for i in range(self.agents):
                y[i][0] = self.agent_y[i][iteration - 1] + self.grads[i][iteration] - self.grads[i][iteration - 1]
            for m in range(self.loop_num):
                for i in range(self.agents):
                    summation_y[i] = np.zeros_like(self.agent_parameters[i][iteration])
                    for j in range(self.agents):
                        summation_y[i] += self.agent_matrix[i, j] * y[j][m]
                    y[i].append(summation_y[i])
            for i in range(self.agents):
                self.agent_y[i].append(y[i][self.loop_num])

            for i in range(self.agents):
                a = la.norm(np.dot((self.agent_parameters[i][iteration] - self.agent_parameters[i][iteration - 1]).T,
                                   (self.agent_parameters[i][iteration] - self.agent_parameters[i][iteration - 1]))) \
                    / la.norm(np.dot((self.agent_parameters[i][iteration] - self.agent_parameters[i][iteration - 1]).T,
                                     (self.grads[i][iteration] - self.grads[i][iteration - 1])))
                b = la.norm(np.dot((self.agent_parameters[i][iteration] - self.agent_parameters[i][iteration - 1]).T,
                                   (self.grads[i][iteration] - self.grads[i][iteration - 1]))) \
                    / la.norm(np.dot((self.grads[i][iteration] - self.grads[i][iteration - 1]).T,
                                     (self.grads[i][iteration] - self.grads[i][iteration - 1])))
                self.eta[i].append(min(a, b))

            for i in range(self.agents):
                x[i][0] = self.agent_parameters[i][iteration] - self.eta[i][iteration] * self.agent_y[i][iteration]
            for m in range(self.loop_num):
                for i in range(self.agents):
                    summation_x[i] = np.zeros_like(self.agent_parameters[i][iteration])
                    for j in range(self.agents):
                        summation_x[i] += self.agent_matrix[i, j] * x[j][m]
                    x[i].append(summation_x[i])
            for i in range(self.agents):
                self.agent_parameters[i].append(x[i][self.loop_num])

    def init_run(self):
        for i in range(self.agents):
            self.eta[i] = [1 / self.agent_L[i]]


class DGD(Trainer):
    def __init__(self, *args, **kwargs):
        super(DGD, self).__init__(*args, **kwargs)
        self.name = "DGD"
        self.eta = {}

    def step(self, iterations):
        for i in range(self.agents):
            summation_x = np.zeros_like(self.agent_parameters[i][iterations])
            for j in range(self.agents):
                summation_x += self.agent_matrix[i, j] * self.agent_parameters[j][iterations]
            self.agent_parameters[i].append(summation_x - self.eta[i][iterations] * self.grads[i][iterations])

            self.eta[i].append(1 / self.agent_L[i])

    def init_run(self):
        for i in range(self.agents):
            self.eta[i] = [1 / self.agent_L[i]]
