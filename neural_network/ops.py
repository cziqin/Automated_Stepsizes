import random

import numpy as np
import torch
from torch.optim.optimizer import Optimizer


class Base(Optimizer):
    def __init__(self, params, idx, w, agents, lr=0.1, num=0.5, kmult=0.007, name=None, device=None,
                 amplifier=.1, theta=np.inf, damping=.4, eps=1e-5, weight_decay=0, kappa=0.9, stratified=True):

        defaults = dict(idx=idx, lr=lr, w=w, num=num, kmult=kmult, agents=agents, name=name, device=device,
                        amplifier=amplifier, theta=theta, damping=damping, eps=eps, weight_decay=weight_decay,
                        kappa=kappa, lamb=lr, stratified=stratified)

        super(Base, self).__init__(params, defaults)

    def collect_prev_params(self, prev_optimizer):
        prev_grads = []
        prev_vars = []
        for prev_group in prev_optimizer.param_groups:
            for prev_p in prev_group['params']:
                if prev_p.grad is None:
                    continue
                prev_vars.append(prev_p.data.clone().detach())
                prev_grads.append(prev_p.grad.data.clone().detach())
        return prev_vars, prev_grads

    def collect_params(self):
        grads = []
        var_new = []
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                var_new.append(p.data.clone().detach())
                grads.append(p.grad.data.clone().detach())
        return var_new, grads

    def collect_lr(self):
        for group in self.param_groups:
            return group["lr"]

    def collect_prev_lr(self):
        for group in self.param_groups:
            return group["prev_lr"]

    def step(self):
        pass


class Algorithm2(Base):
    def __init__(self, *args, **kwargs):
        super(Algorithm2, self).__init__(*args, **kwargs)

    def collect_s(self):
        for group in self.param_groups:
            return group["s"]

    def collect_b(self):
        for group in self.param_groups:
            return group["b"]

    def collect_prev_grads(self):
        for group in self.param_groups:
            return group["prev_grads"]

    def step(self, k=None, vars=None, grads=None, s_all=None, prev_vars=None, prev_grads=None, lr_all=None,
             lr_prev=None, closure=None):
        loss = None

        if closure is not None:
            loss = closure()

        y_norm = 0
        lr_new = 0
        s_list = []
        varsy = {}
        varsx = {}
        lr_a = {}
        for group in self.param_groups:
            idx = group['idx']
            agents = group["agents"]
            device = group["device"]

            for i in range(agents):
                varsy[i] = []
                varsx[i] = []

            if k == 0 or k == 1:
                for j in range(agents):
                    lr_a[j] = lr_all[j]
                lr_new = lr_a[idx]

            sub = 0
            for i, p in enumerate(group['params']):
                if p.grad is None:
                    sub -= 1
                    continue
                summat_y = torch.zeros(p.data.size()).to(device)
                summat_grad = torch.zeros(p.data.size()).to(device)
                for j in range(agents):
                    if k == 0:
                        varsy[j].append(lr_all[j] * grads[j][i + sub])
                    else:
                        varsy[j].append(s_all[j][i + sub].to(device) + lr_all[j] * grads[j][i + sub])
                    summat_y += varsy[j][i + sub]
                    summat_grad += grads[j][i + sub]

                s_list.append(((1 / agents) * summat_y).clone().detach())
                average_grad = summat_grad / agents

                if k == 0:
                    y_norm += (s_list[i + sub] - (lr_all[idx] * average_grad)).norm().item()
                else:
                    y_norm += ((s_list[i + sub] - s_all[idx][i + sub]) - (lr_all[idx] * average_grad)).norm().item()

            sub = 0
            for i, p in enumerate(group['params']):
                if p.grad is None:
                    sub -= 1
                    continue
                summat_x = torch.zeros(p.data.size()).to(device)
                for j in range(agents):
                    if k == 0:
                        varsx[j].append(vars[j][i + sub].to(device) - s_list[i + sub].to(device))
                    else:
                        varsx[j].append(
                            vars[j][i + sub].to(device) - (s_list[i + sub].to(device) - s_all[j][i + sub].to(device)))
                    summat_x += varsx[j][i + sub]
                p.data = (1 / agents) * summat_x

            if k >= 2:
                for j in range(agents):
                    b1 = 0
                    b2 = 0
                    sub = 0
                    for i, p in enumerate(group['params']):
                        if p.grad is None:
                            sub -= 1
                            continue
                        sum_y1 = torch.zeros(p.grad.data.size()).to(device)
                        sum_y2 = torch.zeros(p.grad.data.size()).to(device)
                        for q in range(agents):
                            sum_y1 += lr_all[q] * grads[q][i + sub]
                            sum_y2 += lr_prev[q] * prev_grads[q][i + sub]
                        b1 += (varsx[j][i + sub].to(device) - vars[j][i + sub].to(device)).norm().item() ** 2
                        b2 += (sum_y1 / agents - sum_y2 / agents).norm().item() ** 2
                    a = np.sqrt(1 + lr_all[j] / lr_prev[j]) * lr_all[j]
                    b = np.sqrt(b1) / (2 * np.sqrt(b2))
                    lr_a[j] = min(a, b)
                lr_new = lr_a[idx]
            lr_prev = lr_all[idx]

            group["lr"] = lr_new
            group["prev_lr"] = lr_prev
            group["prev_grads"] = grads[idx]
            group["s"] = s_list

        return y_norm


class AlgorithmS1(Base):
    def __init__(self, *args, **kwargs):
        super(AlgorithmS1, self).__init__(*args, **kwargs)

    def collect_s(self):
        for group in self.param_groups:
            return group["s"]

    def collect_prev_grads(self):
        for group in self.param_groups:
            return group["prev_grads"]

    def step(self, k=None, vars=None, grads=None, s_all=None, prev_vars=None, prev_grads=None, lr_all=None,
             closure=None):
        loss = None

        if closure is not None:
            loss = closure()

        y_norm = 0

        for group in self.param_groups:
            idx = group['idx']
            w = group['w']
            agents = group["agents"]
            device = group["device"]

            s = {}
            varsx = {}
            varsy = {}
            lr_a = {}
            s_list = []
            for i in range(agents):
                s[i] = []
                lr_a[i] = 0

            if k == 0:
                y_norm = 0
                sub = 0
                for i, p in enumerate(group['params']):
                    if p.grad is None:
                        sub -= 1
                        continue
                    for j in range(agents):
                        varsy[j] = [s_all[j][i + sub].to(device)]
                        varsx[j] = [vars[j][i + sub].to(device) - lr_all[j] * varsy[j][0]]

                    summat_x = torch.zeros(p.data.size()).to(device)
                    summat_grad = torch.zeros(p.data.size()).to(device)
                    for j in range(agents):
                        summat_x += w[idx, j] * varsx[j][0]
                        summat_grad += grads[j][i + sub]
                    average_grad = summat_grad / agents
                    p.data = summat_x
                    s_list.append(s_all[idx][i + sub].clone().detach())
                    y_norm += (s_list[i + sub] - average_grad).norm().item()

                lr_new = lr_all[idx]
            else:
                y_norm = 0
                sub = 0
                for i, p in enumerate(group['params']):
                    if p.grad is None:
                        sub -= 1
                        continue
                    for j in range(agents):
                        varsy[j] = [
                            s_all[j][i + sub].to(device) + grads[j][i + sub].to(device) - prev_grads[j][i + sub].to(
                                device)]

                    summat_grad = torch.zeros(p.data.size()).to(device)
                    for j in range(agents):
                        summat_y = torch.zeros(p.data.size()).to(device)
                        for q in range(agents):
                            summat_y += w[j, q] * varsy[q][0]
                        summat_grad += grads[j][i + sub].to(device)
                        s[j].append(summat_y)
                    average_grad = summat_grad / agents
                    s_list.append(s[idx][i + sub].clone().detach())
                    y_norm += (s_list[i + sub] - average_grad).norm().item()

                for j in range(agents):
                    b1 = 0
                    b2 = 0
                    sub = 0
                    for i, p in enumerate(group['params']):
                        if p.grad is None:
                            sub -= 1
                            continue
                        b1 += (vars[j][i + sub].to(device) - prev_vars[j][i + sub].to(device)).norm().item() ** 2
                        b2 += (s[j][i + sub].to(device) - s_all[j][i + sub].to(device)).norm().item() ** 2
                    a = np.sqrt(2) * lr_all[j]
                    b = np.sqrt(b1) / (2 * np.sqrt(b2))
                    c_2 = 1e+10
                    c_1 = 1e-10
                    lr_a[j] = max(min(a, b, c_2), c_1)

                lr_new = lr_a[idx]

                sub = 0
                for i, p in enumerate(group['params']):
                    if p.grad is None:
                        sub -= 1
                        continue
                    for j in range(agents):
                        varsx[j] = [vars[j][i + sub].to(device) - lr_a[j] * s[j][i + sub].to(device)]

                    summat_x = torch.zeros(p.data.size()).to(device)
                    for j in range(agents):
                        summat_x += w[idx, j] * varsx[j][0]

                    p.data = summat_x

            group["lr"] = lr_new
            group["s"] = s_list
            group["prev_grads"] = grads[idx]

        return y_norm


class DSGDN(Base):
    def __init__(self, *args, **kwargs):
        super(DSGDN, self).__init__(*args, **kwargs)

    def collect_u(self):
        for group in self.param_groups:
            return group["u"]

    def step(self, k=None, vars=None, u=None, grads=None, closure=None):
        loss = None
        b = 0.9
        alpha = 0.02

        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            idx = group['idx']
            w = group['w']
            agents = group["agents"]
            device = group["device"]

            u_tilde = {}
            v_tilde = {}
            x_tilde = {}
            u_new = []

            for j in range(agents):
                u_tilde[j], v_tilde[j], x_tilde[j] = [], [], []

            if k == 0:
                for j in range(agents):
                    sub = 0
                    for i, p in enumerate(group['params']):
                        if p.grad is None:
                            sub -= 1
                            continue
                        u[j].append(torch.zeros(p.data.size()).to(device))

            for j in range(agents):
                sub = 0
                for i, p in enumerate(group['params']):
                    if p.grad is None:
                        sub -= 1
                        continue
                    u_tilde[j].append(b * u[j][i + sub].to(device) + (1 - b) * grads[j][i + sub].to(device))
                    v_tilde[j].append(b * u_tilde[j][i + sub].to(device) + grads[j][i + sub].to(device))
                    x_tilde[j].append(vars[j][i + sub].to(device) - alpha * v_tilde[j][i + sub].to(device))

            sub = 0
            for i, p in enumerate(group['params']):
                if p.grad is None:
                    sub -= 1
                    continue
                summat_x = torch.zeros(p.data.size()).to(device)
                summat_u = torch.zeros(p.data.size()).to(device)
                for j in range(agents):
                    summat_x += w[idx, j] * (x_tilde[j][i + sub].to(device))
                    summat_u += w[idx, j] * (u_tilde[j][i + sub].to(device))
                p.data = summat_x
                u_new.append(summat_u)

            lr_new = alpha

            group["u"] = u_new
            group["lr"] = lr_new
        return loss


class DADAM(Base):
    def __init__(self, *args, **kwargs):
        super(DADAM, self).__init__(*args, **kwargs)

    def collect_m(self):
        for group in self.param_groups:
            return group["m"]

    def collect_v(self):
        for group in self.param_groups:
            return group["v"]

    def collect_v_hat(self):
        for group in self.param_groups:
            return group["v_hat"]

    def step(self, k=None, vars=None, m=None, v=None, v_hat=None, closure=None):
        loss = None
        b1 = 0.9
        b2 = 0.99
        epsilon = 1e-3
        alpha = 0.02

        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            idx = group['idx']
            w = group['w']
            agents = group["agents"]
            device = group["device"]

            m_list = []
            v_list = []
            v_hat_list = []

            if k == 0:
                sub = 0
                for i, p in enumerate(group['params']):
                    if p.grad is None:
                        sub -= 1
                        continue
                    m.append(torch.zeros(p.data.size()).to(device))
                    v.append(epsilon * torch.ones_like(torch.mul(p.grad.data, p.grad.data)).to(device))
                    v_hat.append(epsilon * torch.ones_like(torch.mul(p.grad.data, p.grad.data)).to(device))

            lr = 0
            sub = 0
            for i, p in enumerate(group['params']):
                if p.grad is None:
                    sub -= 1
                    continue
                m_new = b1 * m[i + sub].to(device) + (1 - b1) * p.grad.data
                v_new = b2 * v[i + sub].to(device) + (1 - b2) * torch.mul(p.grad.data, p.grad.data)
                v_hat_new = torch.max(v_hat[i + sub].to(device), v_new)
                summat = torch.zeros(p.data.size()).to(device)
                for j in range(agents):
                    summat += w[idx, j] * (vars[j][i + sub].to(device))

                p.data = summat - alpha * (m_new / torch.sqrt(v_hat_new))
                m_list.append(m_new.clone().detach())
                v_list.append(v_new)
                v_hat_list.append(v_hat_new)
                lr += torch.sqrt(v_hat_new).norm().item() ** 2
            lr_new = alpha / np.sqrt(lr)

            group["m"] = m_list
            group["v"] = v_list
            group["v_hat"] = v_hat_list
            group["lr"] = lr_new
        return loss


class DAMSGrad(Base):
    def __init__(self, *args, **kwargs):
        super(DAMSGrad, self).__init__(*args, **kwargs)

    def collect_m(self):
        for group in self.param_groups:
            return group["m"]

    def collect_v(self):
        for group in self.param_groups:
            return group["v"]

    def collect_v_hat(self):
        for group in self.param_groups:
            return group["v_hat"]

    def collect_u_tilde(self):
        for group in self.param_groups:
            return group["u_tilde"]

    def step(self, k=None, vars=None, m=None, v=None, v_hat=None, u_tilde=None, closure=None):
        loss = None
        b1 = 0.9
        b2 = 0.99
        epsilon = 1e-6
        alpha = 0.02

        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            idx = group['idx']
            w = group['w']
            agents = group["agents"]
            device = group["device"]

            m_list = []
            v_list = []
            v_hat_list = []
            u_tilde_list = []

            if k == 0:
                sub = 0
                for i, p in enumerate(group['params']):
                    if p.grad is None:
                        sub -= 1
                        continue
                    m.append(torch.zeros(p.data.size()).to(device))
                    v.append(epsilon * torch.ones_like(torch.mul(p.grad.data, p.grad.data)).to(device))
                    v_hat.append(epsilon * torch.ones_like(torch.mul(p.grad.data, p.grad.data)).to(device))
                    for j in range(agents):
                        u_tilde[j].append(epsilon * torch.ones_like(torch.mul(p.grad.data, p.grad.data)).to(device))

            lr = 0
            sub = 0
            for i, p in enumerate(group['params']):
                if p.grad is None:
                    sub -= 1
                    continue
                m_new = b1 * m[i + sub].to(device) + (1 - b1) * p.grad.data
                v_new = b2 * v[i + sub].to(device) + (1 - b2) * torch.mul(p.grad.data, p.grad.data)
                v_hat_new = torch.max(v_hat[i + sub].to(device), v_new)
                summat_x = torch.zeros(p.data.size()).to(device)
                summat_u = torch.zeros_like(torch.mul(p.grad.data, p.grad.data))
                for j in range(agents):
                    summat_x += w[idx, j] * vars[j][i + sub].to(device)
                    summat_u += w[idx, j] * u_tilde[j][i + sub].to(device)
                x_temp = summat_x
                u_temp = summat_u
                u_new = torch.max(u_temp,
                                  epsilon * torch.ones_like(torch.mul(p.grad.data, p.grad.data)))

                p.data = x_temp - alpha * (m_new / torch.sqrt(u_new))
                u_tilde_list.append(u_temp - v_hat[i + sub].to(device) + v_hat_new)
                m_list.append(m_new.clone().detach())
                v_list.append(v_new.clone().detach())
                v_hat_list.append(v_hat_new.clone().detach())
                lr += torch.sqrt(u_new).norm().item() ** 2
            lr_new = alpha / np.sqrt(lr)

            group["u_tilde"] = u_tilde_list
            group["m"] = m_list
            group["v"] = v_list
            group["v_hat"] = v_hat_list
            group["lr"] = lr_new
        return loss


