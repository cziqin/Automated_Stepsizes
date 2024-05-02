import csv

import numpy as np
import scipy.linalg as LA
# from scipy.sparse import random as r
import numpy.linalg as la
import matplotlib.pyplot as plt
import math


def func(param, A):
    m, n = A.shape
    U, V = param[:m], param[m:]
    return 0.5 * LA.norm(U @ V.T - A) ** 2


def cgrad(param, A):
    m, n = A.shape
    U, V = param[:m], param[m:]
    res = U @ V.T - A
    grad_U = res @ V
    grad_V = res.T @ U
    return np.vstack([grad_U, grad_V])


def safe_division(x, y):
    return np.exp(np.log(x) - np.log(y)) if y != 0 else 1e4


def Algorithm1(w, A, cgrad, L, agents=10, iterations=2000, skip=200):
    name = "Algorithm1"
    param_dict = {}
    y_dict = {}
    grad_dict = {}
    loss_dict = {}
    eta_dict = {}
    varx = {}
    vary = {}
    row, col = A[0].shape
    r = 10

    np.random.seed(0)
    for j in range(agents):
        param_dict[j] = [np.random.randn(row + col, r)]
        eta_dict[j] = [1 / L]

    iters = []
    loss_value = []
    grad_norms = []
    average_eta = []
    for t in range(iterations):
        if t == 0:
            sum_x = np.zeros_like(param_dict[0][t])
            for i in range(agents):
                varx[i] = []
                grad_dict[i] = [cgrad(param_dict[i][t], A[i])]
                loss_dict[i] = [func(param_dict[i][t], A[i])]
                y_dict[i] = [grad_dict[i][t]]

                varx[i].append(param_dict[i][t] - eta_dict[i][t] * y_dict[i][t])

            for i in range(agents):
                sum_x += varx[i][0]

            for i in range(agents):
                param_dict[i].append(sum_x / agents)
        else:

            sum_y = np.zeros_like(grad_dict[0][t - 1])
            for i in range(agents):
                vary[i] = []
                grad_dict[i].append(cgrad(param_dict[i][t], A[i]))
                loss_dict[i].append(func(param_dict[i][t], A[i]))

                vary[i].append(y_dict[i][t - 1] + grad_dict[i][t] - grad_dict[i][t - 1])
            for i in range(agents):
                sum_y += vary[i][0]

            for i in range(agents):
                y_dict[i].append(sum_y / agents)


            if t == 1:
                for i in range(agents):
                    eta_dict[i].append(eta_dict[i][0])
            else:
                for i in range(agents):
                    a = np.sqrt(1 + eta_dict[i][t - 1] / eta_dict[i][t - 2]) * eta_dict[i][t - 1]
                    b = la.norm(param_dict[i][t] - param_dict[i][t - 1]) / (
                            2.0 * la.norm(y_dict[i][t] - y_dict[i][t - 1]))
                    eta_dict[i].append(min(a, b))

            sum_x = np.zeros_like(param_dict[0][t])
            for i in range(agents):
                varx[i] = []
                varx[i].append(param_dict[i][t] - eta_dict[i][t] * y_dict[i][t])
            for i in range(agents):
                sum_x += varx[i][0]

            for i in range(agents):
                param_dict[i].append(sum_x / agents)

        iters.append(t)
        grad_norms.append(np.linalg.norm(sum([grad[t] for grad in grad_dict.values()]) / agents))
        loss_value.append(np.sum([loss[t] for loss in loss_dict.values()]) / agents)

        average_eta.append(sum([eta[t] for eta in eta_dict.values()]) / agents)
        if t % skip == 0:
            print(
                f"{name}, Iteration: {t}, Gradient Norm: {grad_norms[t]}, Eta: {average_eta[t]}")

    return iters, grad_norms, name, average_eta


def Algorithms1(w, A, cgrad, L, agents=10, iterations=2000, skip=200):
    name = "Algorithms1"
    param_dict = {}
    y_dict = {}
    grad_dict = {}
    loss_dict = {}
    eta_dict = {}
    varx = {}
    vary = {}
    row, col = A[0].shape
    r = 10

    np.random.seed(0)
    for j in range(agents):
        param_dict[j] = [np.random.randn(row + col, r)]
        eta_dict[j] = [1 / L]

    iters = []
    grad_norms = []
    average_eta = []
    loss_value = []
    for t in range(iterations):
        if t == 0:
            for i in range(agents):
                varx[i] = []
                grad_dict[i] = [cgrad(param_dict[i][t], A[i])]
                loss_dict[i] = [func(param_dict[i][t], A[i])]
                y_dict[i] = [grad_dict[i][t]]

                varx[i].append(param_dict[i][t] - eta_dict[i][t] * y_dict[i][t])

            for i in range(agents):
                sum_x = np.zeros_like(param_dict[0][t])
                for j in range(agents):
                    sum_x += w[i, j] * varx[j][0]
                param_dict[i].append(sum_x)
        else:
            for i in range(agents):
                vary[i] = []
                grad_dict[i].append(cgrad(param_dict[i][t], A[i]))
                loss_dict[i].append(func(param_dict[i][t], A[i]))
                vary[i].append(y_dict[i][t - 1] + grad_dict[i][t] - grad_dict[i][t - 1])

            for i in range(agents):
                sum_y = np.zeros_like(grad_dict[0][t])
                for j in range(agents):
                    sum_y += w[i, j] * vary[j][0]
                y_dict[i].append(sum_y)

            if t == 1:
                for i in range(agents):
                    eta_dict[i].append(eta_dict[i][0])
            else:
                for i in range(agents):
                    a = np.sqrt(2) * eta_dict[i][t - 1]
                    b = la.norm(param_dict[i][t] - param_dict[i][t - 1]) / (
                            2.0 * la.norm(y_dict[i][t] - y_dict[i][t - 1]))
                    c_2 = 1e+10
                    c_1 = 1e-10
                    eta_dict[i].append(min(a, b))

            for i in range(agents):
                varx[i] = []
                varx[i].append(param_dict[i][t] - eta_dict[i][t] * y_dict[i][t])

            for i in range(agents):
                sum_x = np.zeros_like(param_dict[0][t])
                for j in range(agents):
                    sum_x += w[i, j] * varx[j][0]
                param_dict[i].append(sum_x)

        iters.append(t)
        grad_norms.append(np.linalg.norm(sum([grad[t] for grad in grad_dict.values()]) / agents))
        loss_value.append(np.sum([loss[t] for loss in loss_dict.values()]) / agents)

        average_eta.append(sum([eta[t] for eta in eta_dict.values()]) / agents)
        if t % skip == 0:
            print(
                f"{name}, Iteration: {t}, Gradient Norm: {grad_norms[t]}, Eta: {average_eta[t]}")

    return iters, grad_norms, name, average_eta


def DBBG(w, A, cgrad, L, agents=10, iterations=2000, skip=200):
    name = "DBBG"
    param_dict = {}
    y_dict = {}
    grad_dict = {}
    loss_dict = {}
    eta_dict = {}
    varx = {}
    vary = {}
    # Choose the value for r (10, 20, 30)
    row, col = A[0].shape
    r = 10

    np.random.seed(0)
    for j in range(agents):
        param_dict[j] = [np.random.randn(row + col, r)]
        eta_dict[j] = [1 / L]
    iters = []
    loss_value = []
    grad_norms = []
    average_eta = []
    loss_value = []
    for t in range(iterations):
        if t == 0:
            for i in range(agents):
                varx[i] = []
                grad_dict[i] = [cgrad(param_dict[i][t], A[i])]
                loss_dict[i] = [func(param_dict[i][t], A[i])]
                y_dict[i] = [grad_dict[i][t]]
                varx[i].append(param_dict[i][t] - eta_dict[i][t] * y_dict[i][t])

            for i in range(agents):
                sum_x = np.zeros_like(param_dict[0][t])
                for j in range(agents):
                    sum_x += w[i, j] * varx[j][0]
                param_dict[i].append(sum_x)
        else:
            for i in range(agents):
                vary[i] = []
                grad_dict[i].append(cgrad(param_dict[i][t], A[i]))
                loss_dict[i].append(func(param_dict[i][t], A[i]))
                vary[i].append(y_dict[i][t - 1] + grad_dict[i][t] - grad_dict[i][t - 1])

            for i in range(agents):
                sum_y = np.zeros_like(grad_dict[0][t])
                for j in range(agents):
                    sum_y += w[i, j] * vary[j][0]
                y_dict[i].append(sum_y)

            if t == 1:
                for i in range(agents):
                    eta_dict[i].append(eta_dict[i][0])
            else:
                for i in range(agents):
                    a = la.norm(
                        np.dot((param_dict[i][t] - param_dict[i][t - 1]).T,
                               (param_dict[i][t] - param_dict[i][t - 1]))) \
                        / la.norm(
                        np.dot((param_dict[i][t] - param_dict[i][t - 1]).T,
                               (grad_dict[i][t] - grad_dict[i][t - 1])))
                    b = la.norm(
                        np.dot((param_dict[i][t] - param_dict[i][t - 1]).T,
                               (grad_dict[i][t] - grad_dict[i][t - 1]))) \
                        / la.norm(np.dot((grad_dict[i][t] - grad_dict[i][t - 1]).T,
                                         (grad_dict[i][t] - grad_dict[i][t - 1])))
                    eta_dict[i].append(min(a, b))

            for i in range(agents):
                varx[i] = []
                varx[i].append(param_dict[i][t] - eta_dict[i][t] * y_dict[i][t])

            for i in range(agents):
                sum_x = np.zeros_like(param_dict[0][t])
                for j in range(agents):
                    sum_x += w[i, j] * varx[j][0]
                param_dict[i].append(sum_x)

        iters.append(t)
        grad_norms.append(np.linalg.norm(sum([grad[t] for grad in grad_dict.values()]) / agents))
        loss_value.append(np.sum([loss[t] for loss in loss_dict.values()]) / agents)

        average_eta.append(sum([eta[t] for eta in eta_dict.values()]) / agents)
        if t % skip == 0:
            print(
                f"{name}, Iteration: {t}, Gradient Norm: {grad_norms[t]}, Eta: {average_eta[t]}")

    return iters, grad_norms, name, average_eta


def CDGDP(w, A, cgrad, L, agents=10, iterations=2000, skip=200):
    name = "CDGDP"
    param_dict = {}
    y_dict = {}
    grad_dict = {}
    loss_dict = {}
    eta_dict = {}
    tilde_u = {}
    tilde_x = {}
    u = {}
    row, col = A[0].shape
    r = 10
    b = 0.3
    gamma = np.s / L

    np.random.seed(0)
    for i in range(agents):
        param_dict[i] = [np.random.randn(row + col, r)]
        eta_dict[i] = [gamma]

    iters = []
    loss_value = []
    grad_norms = []
    average_eta = []
    loss_value = []
    for t in range(iterations):
        if t == 0:
            for i in range(agents):
                grad_dict[i] = [cgrad(param_dict[i][t], A[i])]
                loss_dict[i] = [func(param_dict[i][t], A[i])]
                u[i] = [np.zeros_like(grad_dict[0][t])]

                tilde_u[i] = b * u[i][t] + grad_dict[i][t]
                tilde_x[i] = param_dict[i][t] - gamma * tilde_u[i]
            for i in range(agents):
                sum_u = np.zeros_like(grad_dict[0][t])
                sum_x = np.zeros_like(param_dict[0][t])
                for j in range(agents):
                    sum_u += w[i, j] * tilde_u[j]
                    sum_x += w[i, j] * tilde_x[j]
                param_dict[i].append(sum_x)
                u[i].append(sum_u)
        else:
            for i in range(agents):
                grad_dict[i].append(cgrad(param_dict[i][t], A[i]))
                loss_dict[i].append(func(param_dict[i][t], A[i]))

                tilde_u[i] = b * u[i][t] + grad_dict[i][t]
                tilde_x[i] = param_dict[i][t] - gamma * tilde_u[i]
            for i in range(agents):
                sum_u = np.zeros_like(grad_dict[0][t])
                sum_x = np.zeros_like(param_dict[0][t])
                for j in range(agents):
                    sum_u += w[i, j] * tilde_u[j]
                    sum_x += w[i, j] * tilde_x[j]
                param_dict[i].append(sum_x)
                u[i].append(sum_u)

                eta_dict[i].append(gamma)

        iters.append(t)
        grad_norms.append(np.linalg.norm(sum([grad[t] for grad in grad_dict.values()]) / agents))
        loss_value.append(np.sum([loss[t] for loss in loss_dict.values()]) / agents)

        average_eta.append(sum([eta[t] for eta in eta_dict.values()]) / agents)
        if t % skip == 0:
            print(
                f"{name}, Iteration: {t}, Gradient Norm: {grad_norms[t]}, Eta: {average_eta[t]}")

    return iters, grad_norms, name, average_eta


def record_results(filename, optimizers, skip):
    for i, opt in enumerate(optimizers):
        with open(filename, mode="a") as csv_file:
            file = csv.writer(csv_file, lineterminator="\n")
            file.writerow([opt[2]])
            file.writerow(opt[0][::skip])
            file.writerow(opt[1][::skip])
            file.writerow(opt[3][::skip])
