import os
import pickle
from collections import OrderedDict
import copy
import csv
from random import shuffle, sample
import random
from time import perf_counter
import warnings
import math
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

from models import *
from ops import Algorithm2, AlgorithmS1, DSGDN, DADAM, DAMSGrad
from util import FixedQueue

warnings.filterwarnings("ignore")


class DTrainer:
    def __init__(self,
                 dataset="mnist",
                 epochs=6,
                 batch_size=40,
                 lr=0.02,
                 workers=4,
                 agents=10,
                 num=0.1,
                 kmult=0.5,
                 exp=0.7,
                 w=None,
                 kappa=0.9,
                 fname=None,
                 stratified=True):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_accuracy = []
        self.test_accuracy = []
        self.train_iterations = []
        self.test_iterations = []
        self.lr_logs = []
        self.lr_actual_logs = []
        self.noise_logs = []
        self.y_norm = {}
        self.loss_list = []

        self.dataset = dataset
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.workers = workers
        self.agents = agents
        self.num = num
        self.kmult = kmult
        self.running_iteration = 0
        self.exp = exp
        self.kappa = kappa
        self.fname = fname
        self.stratified = stratified
        self.load_data()
        self.w = w
        self.criterion = torch.nn.CrossEntropyLoss()
        self.agent_setup()

    def _log(self, accuracy):
        """ Helper function to log accuracy values"""
        self.train_accuracy.append(accuracy)
        self.train_iterations.append(self.running_iteration)

    def _save(self):
        with open(self.fname, mode='a') as csv_file:
            file = csv.writer(csv_file, lineterminator='\n')
            file.writerow([f"{self.opt_name}, {self.num}, {self.kmult}, {self.batch_size}, {self.epochs}"])
            file.writerow(self.train_iterations)
            file.writerow(self.train_accuracy)
            file.writerow(self.test_accuracy)
            file.writerow(self.loss_list)
            file.writerow(["ETA"])
            file.writerow(self.lr_logs)
            if self.opt_name == "Algorithm1":
                file.writerow(["ETA_actual"])
                file.writerow(self.lr_actual_logs)
            elif self.opt_name == "AlgorithmS1" or self.opt_name == "Algorithm2" or self.opt_name == "Algorithm1":
                file.writerow(["Noise"])
                file.writerow(self.noise_logs)
            file.writerow([])
            file.writerow([])

    def load_data(self):
        print("==> Loading Data")
        self.train_loader = {}
        self.test_loader = {}

        if self.dataset == 'cifar10':
            transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                                  transforms.RandomHorizontalFlip(),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                       (0.2023, 0.1994, 0.2010)), ])

            transform_test = transforms.Compose([transforms.ToTensor(),
                                                 transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                      (0.2023, 0.1994, 0.2010)), ])
            self.class_num = 10
            trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
            testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

        elif self.dataset == "mnist":
            transform_train = transforms.Compose([transforms.ToTensor(), ])
            transform_test = transforms.Compose([transforms.ToTensor(), ])

            self.class_num = 10
            trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_train)
            testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_test)
        else:
            raise ValueError(f'{self.dataset} is not supported')

        if self.stratified:
            train_len, test_len = int(len(trainset)), int(len(testset))

            temp_train = torch.utils.data.random_split(trainset, [int(train_len // self.agents)] * self.agents)

            for i in range(self.agents):
                self.train_loader[i] = torch.utils.data.DataLoader(temp_train[i], batch_size=self.batch_size,
                                                                   shuffle=True)

            self.test_loader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False)
        else:
            train_len, test_len = int(len(trainset)), int(len(testset))
            idxs = {}
            for i in range(0, 10, 1):
                arr = np.array(trainset.targets,
                               dtype=int)
                idxs[int(i)] = list(np.where(arr == i)[
                                        0])
                shuffle(idxs[int(i)])
            percent_main = 0.3
            percent_else = (1 - percent_main) / (self.agents - 1)
            main_samp_num = int(percent_main * len(idxs[5]))
            sec_samp_num = int(percent_else * len(idxs[5]))

            for i in range(self.agents):
                agent_idxs = []
                for j in range(self.agents):
                    if i == j:
                        agent_idxs.extend(sample(idxs[j], main_samp_num))
                    else:
                        agent_idxs.extend(sample(idxs[j], sec_samp_num))
                    idxs[j] = list(filter(lambda x: x not in agent_idxs, idxs[j]))
                temp_train = copy.deepcopy(trainset)
                temp_train.targets = [temp_train.targets[i] for i in agent_idxs]
                temp_train.data = [temp_train.data[i] for i in agent_idxs]
                self.train_loader[i] = torch.utils.data.DataLoader(temp_train, batch_size=self.batch_size, shuffle=True)
            self.test_loader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False)


    def agent_setup(self):
        for i in range(self.agents):
            if self.opt_name == "Algorithm1" or self.opt_name == "Algorithm2" or self.opt_name == "AlgorithmS1":
                self.y_norm[i] = []
            if self.opt_name == "Algorithm1":
                self.lr_actual_logs[i] = []

        self.agent_models = {}
        self.prev_agent_models = {}
        self.agent_optimizers = {}
        self.prev_agent_optimizers = {}
        model = None

        if self.dataset == 'cifar10':
            model = CIFAR10CNN()

        elif self.dataset == "imagenet":
            raise ValueError("ImageNet Not Supported: Low Computing Power")

        elif self.dataset == "mnist":
            model = MnistCNN()

        for i in range(self.agents):
            if i == 0:
                if int(torch.cuda.device_count()) > 1:
                    self.agent_models[i] = torch.nn.DataParallel(model)
                else:
                    self.agent_models[i] = model

            else:
                if int(torch.cuda.device_count()) > 1:
                    self.agent_models[i] = copy.deepcopy(self.agent_models[0])
                else:
                    self.agent_models[i] = copy.deepcopy(model)

            # # TODO:change the initialization of the model
            # import pickle
            # with open(f'epoch_4__iteration_350/agent_{i}_model_state_dict.pkl', 'rb') as f:
            #     state = pickle.load(f)
            # self.agent_models[i] = copy.deepcopy(model)
            # self.agent_models[i].load_state_dict(state)

            self.agent_models[i].to(self.device)
            self.agent_models[i].train()

            if self.opt_name == "AlgorithmS1" or self.opt_name == "DBBG" or self.opt_name == "Algorithm1" or self.opt_name == "Algorithm2" or self.opt_name == "Centralized":
                self.prev_agent_models[i] = copy.deepcopy(model)
                self.prev_agent_models[i].to(self.device)
                self.prev_agent_models[i].train()
                self.prev_agent_optimizers[i] = self.opt(
                    params=self.prev_agent_models[i].parameters(),
                    idx=i,
                    w=self.w,
                    agents=self.agents,
                    lr=self.lr,
                    num=self.num,
                    kmult=self.kmult,
                    name=self.opt_name,
                    device=self.device,
                    kappa=self.kappa,
                    stratified=self.stratified
                )

            self.agent_optimizers[i] = self.opt(
                params=self.agent_models[i].parameters(),
                idx=i,
                w=self.w,
                agents=self.agents,
                lr=self.lr,
                num=self.num,
                kmult=self.kmult,
                name=self.opt_name,
                device=self.device,
                kappa=self.kappa,
                stratified=self.stratified
            )

    def eval(self, dataloader):
        total_acc, total_count = 0, 0

        with torch.no_grad():

            for i in range(self.agents):
                self.agent_models[i].eval()

                for inputs, labels in dataloader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    predicted_label = self.agent_models[i](inputs)

                    total_acc += (predicted_label.argmax(1) == labels).sum().item()
                    total_count += labels.size(0)

        self.test_iterations.append(self.running_iteration)
        self.test_accuracy.append(total_acc / total_count)

        return total_acc / total_count

    def it_logger(self, total_acc, total_count, epoch, log_interval, tot_loss, start_time):
        self._log(total_acc / total_count)
        t_acc = self.eval(self.test_loader)

        total_sum_step = 0
        for i in range(self.agents):
            total_sum_step += self.agent_optimizers[i].collect_lr()
        average_step = total_sum_step / self.agents
        self.lr_logs.append(average_step)

        if self.opt_name == "Algorithm1":
            total_sum_step_b = 0
            total_y_norm = 0
            for i in range(self.agents):
                total_sum_step_b += self.lr_actual_logs[i][-1]
                total_y_norm += self.y_norm[i][-1]
            average_step_actual = total_sum_step_b / self.agents
            average_y = total_y_norm / self.agents
            self.lr_actual_logs.append(average_step_actual)
            self.noise_logs.append(average_y)

            print(
                f"Epoch: {epoch + 1}, Iteration: {self.running_iteration}, " +
                f"Accuracy: {total_acc / total_count:.4f}, " +
                f"Test Accuracy: {t_acc:.4f}, " +
                f"Loss: {tot_loss / (self.agents * log_interval):.4f}, " +
                f"average_y: {average_y:.4f}, " +
                f"stepsize: {average_step:.4f}, " +
                f"stepsize_atcual: {average_step_actual:.4f}, " +
                f"Time taken: {perf_counter() - start_time:.4f}"
            )


        elif self.opt_name == "AlgorithmS1"  or self.opt_name == "Algorithm2":
            total_y_norm = 0
            for i in range(self.agents):
                total_y_norm += self.y_norm[i][-1]
            average_y = total_y_norm / self.agents
            self.noise_logs.append(average_y)

            print(
                f"Epoch: {epoch + 1}, Iteration: {self.running_iteration}, " +
                f"Accuracy: {total_acc / total_count:.4f}, " +
                f"Test Accuracy: {t_acc:.4f}, " +
                f"Loss: {tot_loss / (self.agents * log_interval):.4f}, " +
                f"average_y: {average_y:.4f}, " +
                f"stepsize: {average_step:.4f}, " +
                f"Time taken: {perf_counter() - start_time:.4f}"
            )


        else:
            print(
                f"Epoch: {epoch + 1}, Iteration: {self.running_iteration}, " +
                f"Accuracy: {total_acc / total_count:.4f}, " +
                f"Test Accuracy: {t_acc:.4f}, " +
                f"Loss: {tot_loss / (self.agents * log_interval):.4f}, " +
                f"stepsize: {average_step:.4f}, " +
                f"Time taken: {perf_counter() - start_time:.4f}"
            )


        self.loss_list.append(tot_loss / (self.agents * log_interval))

    def trainer(self):
        if self.opt_name == "AlgorithmS1" or self.opt_name == "Algorithm2" or self.opt_name == "Algorithm1":
            print(
                f"==> Starting Training for {self.opt_name}, {self.epochs} epochs and {self.agents} agents on the {self.dataset} dataset, via {self.device}")
        else:
            print(
                f"==> Starting Training for {self.opt_name}, {self.epochs} epochs and {self.agents} agents on the {self.dataset} dataset, via {self.device}" +
                f" for {self.num}, {self.kmult}")
        for i in range(self.agents):
            self.test_accuracy = []
            self.train_accuracy = []

        for i in range(self.epochs):
            self.epoch_iterations(i, self.train_loader)


class Algorithm2Trainer(DTrainer):
    def __init__(self, *args, **kwargs):
        self.opt = Algorithm2
        self.opt_name = "Algorithm2"
        super().__init__(*args, **kwargs)

        self.trainer()
        self._save()

    def epoch_iterations(self, epoch,
                         dataloader):
        start_time = perf_counter()
        if self.dataset == "cifar10":
            log_interval = 19
        else:
            log_interval = 14

        loss, prev_loss = {}, {}
        total_acc, total_count, tot_loss = 0, 0, 0

        for idx, data in enumerate(zip(*dataloader.values())):
            self.running_iteration = idx + epoch * len(dataloader[0])
            vars, grads, s, prev_grads, prev_vars, lr, prev_lr, b = {}, {}, {}, {}, {}, {}, {}, {}

            seed = 52
            random.seed(seed)
            os.environ['PYTHONHASHSEED'] = str(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True

            average_grads_dict = {}
            average_prev_grads_dict = {}
            for i in range(self.agents):
                inputs, labels = data[i]
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.prev_agent_models[i].train()

                self.prev_agent_optimizers[i].zero_grad()
                prev_predicted_label = self.prev_agent_models[i](inputs)
                prev_loss[i] = self.criterion(prev_predicted_label, labels)
                prev_loss[i].backward()
                prev_vars[i] = self.agent_optimizers[i].collect_prev_params(
                    self.prev_agent_optimizers[i])


                self.agent_optimizers[i].zero_grad()
                self.agent_models[i].train()
                predicted_label = self.agent_models[i](inputs)
                loss[i] = self.criterion(predicted_label, labels)
                loss[i].backward()
                vars[i], grads[i] = self.agent_optimizers[i].collect_params()

                if self.running_iteration == 0:
                    s[i] = 0
                    # b[i] = 0
                    lr[i] = self.lr
                    prev_lr[i] = self.lr
                    prev_grads[i] = 0
                else:
                    s[i] = self.agent_optimizers[i].collect_s()
                    lr[i] = self.agent_optimizers[i].collect_lr()
                    # b[i] = self.agent_optimizers[i].collect_b()
                    prev_lr[i] = self.agent_optimizers[i].collect_prev_lr()
                    prev_grads[i] = self.agent_optimizers[i].collect_prev_grads()

                if torch.cuda.device_count() > 1:
                    new_mod_state_dict = OrderedDict()
                    for k, v in self.agent_models[i].state_dict().items():
                        new_mod_state_dict[k[7:]] = v
                    self.prev_agent_models[i].load_state_dict(new_mod_state_dict)
                else:
                    self.prev_agent_models[i].load_state_dict(self.agent_models[i].state_dict())

                total_acc += (predicted_label.argmax(1) == labels).sum().item()
                total_count += labels.size(0)

                tot_loss += loss[i].item()

            for i in range(self.agents):
                # TODO: add dict params,average grads and average prev_grads
                y_norm = self.agent_optimizers[i].step(self.running_iteration, vars=vars, grads=grads, s_all=s,
                                                       prev_vars=prev_vars, prev_grads=prev_grads,
                                                       lr_all=lr, lr_prev=prev_lr)
                self.y_norm[i].append(y_norm)
            if 0 == idx % log_interval and epoch % 1 == 0:
                if (epoch >= 10) and (self.running_iteration % 50 == 0):
                    models_save_path = f"epoch_{epoch}__iteration_{self.running_iteration}"
                    if not os.path.exists(models_save_path):
                        os.makedirs(models_save_path)
                    for agent in range(self.agents):
                        with open(f"{models_save_path}/agent_{agent}_model_state_dict.pkl", "wb") as f:
                            pickle.dump(self.agent_models[agent].state_dict(), f)
                self.it_logger(total_acc, total_count, epoch, log_interval, tot_loss, start_time)
                total_acc, total_count, tot_loss = 0, 0, 0
                for i in range(self.agents):
                    self.agent_models[i].train()
                start_time = perf_counter()
        return total_acc


class AlgorithmS1Trainer(DTrainer):
    def __init__(self, *args, **kwargs):
        self.opt = AlgorithmS1
        self.opt_name = "AlgorithmS1"
        super().__init__(*args, **kwargs)
        self.trainer()
        self._save()

    def epoch_iterations(self, epoch,
                         dataloader):
        start_time = perf_counter()
        if self.dataset == "cifar10":
            log_interval = 7
            log_interval = 14

        loss, prev_loss = {}, {}
        total_acc, total_count, tot_loss = 0, 0, 0

        for idx, data in enumerate(zip(*dataloader.values())):
            self.running_iteration = idx + epoch * len(dataloader[0])
            vars, grads, s, prev_grads, prev_vars, lr = {}, {}, {}, {}, {}, {}

            for i in range(self.agents):
                self.agent_optimizers[i].zero_grad(set_to_none=True)
                self.prev_agent_optimizers[i].zero_grad(set_to_none=True)
                inputs, labels = data[i]
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                prev_predicted_label = self.prev_agent_models[i](inputs)
                prev_loss[i] = self.criterion(prev_predicted_label, labels)
                prev_loss[i].backward()
                prev_vars[i], prev_grads[i] = self.prev_agent_optimizers[i].collect_params()

                predicted_label = self.agent_models[i](inputs)
                loss[i] = self.criterion(predicted_label, labels)
                loss[i].backward()
                vars[i], grads[i] = self.agent_optimizers[i].collect_params()

                if torch.cuda.device_count() > 1:
                    new_mod_state_dict = OrderedDict()

                    for k, v in self.agent_models[i].state_dict().items():
                        new_mod_state_dict[k[7:]] = v
                    self.prev_agent_models[i].load_state_dict(new_mod_state_dict)
                else:
                    self.prev_agent_models[i].load_state_dict(self.agent_models[i].state_dict())

                if self.running_iteration == 0:
                    s[i] = grads[i]
                    lr[i] = self.lr

                else:
                    s[i] = self.agent_optimizers[i].collect_s()
                    lr[i] = self.agent_optimizers[i].collect_lr()

                total_acc += (predicted_label.argmax(1) == labels).sum().item()
                total_count += labels.size(0)

                tot_loss += loss[i].item()

            for i in range(self.agents):
                y_norm = self.agent_optimizers[i].step(self.running_iteration, vars=vars, grads=grads, s_all=s,
                                                       prev_vars=prev_vars, prev_grads=prev_grads,
                                                       lr_all=lr)
                self.y_norm[i].append(y_norm)

            if idx % log_interval == 0 and epoch % 1 == 0:
                self.it_logger(total_acc, total_count, epoch, log_interval, tot_loss, start_time)
                total_acc, total_count, tot_loss = 0, 0, 0
                for i in range(self.agents):
                    self.agent_models[i].train()
                start_time = perf_counter()
        return total_acc


class DSGDNTrainer(DTrainer):
    def __init__(self, *args, **kwargs):
        self.opt = DSGDN
        self.opt_name = "DSGDN"
        super().__init__(*args, **kwargs)
        self.trainer()
        self._save()

    def epoch_iterations(self, epoch,
                         dataloader):
        start_time = perf_counter()
        if self.dataset == "cifar10":
            log_interval = 7
        else:
            log_interval = 14

        loss, prev_loss = {}, {}
        total_acc, total_count, tot_loss = 0, 0, 0

        for idx, data in enumerate(zip(*dataloader.values())):
            self.running_iteration = idx + epoch * len(dataloader[0])
            vars, grads, u, lr = {}, {}, {}, {}

            for i in range(self.agents):
                self.agent_optimizers[i].zero_grad()
                inputs, labels = data[i]
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                predicted_label = self.agent_models[i](inputs)
                loss[i] = self.criterion(predicted_label, labels)
                loss[i].backward()
                vars[i], grads[i] = self.agent_optimizers[i].collect_params()

                if self.running_iteration == 0:
                    u[i] = []
                else:
                    u[i] = self.agent_optimizers[i].collect_u()

                total_acc += (predicted_label.argmax(1) == labels).sum().item()
                total_count += labels.size(0)

                tot_loss += loss[i].item()

            for i in range(self.agents):
                self.agent_optimizers[i].step(self.running_iteration, vars=vars, u=u, grads=grads)

            if idx % log_interval == 0 and epoch % 1 == 0:
                self.it_logger(total_acc, total_count, epoch, log_interval, tot_loss, start_time)
                total_acc, total_count, tot_loss = 0, 0, 0
                for i in range(self.agents):
                    self.agent_models[i].train()
                start_time = perf_counter()
        return total_acc


class DADAMTrainer(DTrainer):
    def __init__(self, *args, **kwargs):
        self.opt = DADAM
        self.opt_name = "DADAM"
        super().__init__(*args, **kwargs)
        self.trainer()
        self._save()

    def epoch_iterations(self, epoch,
                         dataloader):
        start_time = perf_counter()
        if self.dataset == "cifar10":
            log_interval = 19
        else:
            log_interval = 14

        loss, prev_loss = {}, {}
        total_acc, total_count, tot_loss = 0, 0, 0

        for idx, data in enumerate(zip(*dataloader.values())):
            self.running_iteration = idx + epoch * len(dataloader[0])
            vars, grads, m, v, v_hat, lr = {}, {}, {}, {}, {}, {}

            seed = 42
            random.seed(seed)
            os.environ['PYTHONHASHSEED'] = str(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

            torch.backends.cudnn.deterministic = True

            for i in range(self.agents):
                self.agent_optimizers[i].zero_grad()
                inputs, labels = data[i]
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                predicted_label = self.agent_models[i](inputs)
                loss[i] = self.criterion(predicted_label, labels)
                loss[i].backward()
                vars[i], grads[i] = self.agent_optimizers[i].collect_params()

                if self.running_iteration == 0:
                    m[i] = []
                    v[i] = []
                    v_hat[i] = []
                else:
                    m[i] = self.agent_optimizers[i].collect_m()
                    v[i] = self.agent_optimizers[i].collect_v()
                    v_hat[i] = self.agent_optimizers[i].collect_v_hat()

                total_acc += (predicted_label.argmax(1) == labels).sum().item()
                total_count += labels.size(0)

                tot_loss += loss[i].item()

            for i in range(self.agents):
                self.agent_optimizers[i].step(self.running_iteration, vars=vars, m=m[i], v=v[i],
                                              v_hat=v_hat[i])
            if idx % log_interval == 0 and epoch % 1 == 0:
                self.it_logger(total_acc, total_count, epoch, log_interval, tot_loss, start_time)
                total_acc, total_count, tot_loss = 0, 0, 0
                for i in range(self.agents):
                    self.agent_models[i].train()
                start_time = perf_counter()
        return total_acc


class DAMSGradTrainer(DTrainer):
    def __init__(self, *args, **kwargs):
        self.opt = DAMSGrad
        self.opt_name = "DAMSGrad"
        super().__init__(*args, **kwargs)
        self.trainer()
        self._save()

    def epoch_iterations(self, epoch,
                         dataloader):
        start_time = perf_counter()
        if self.dataset == "cifar10":
            log_interval = 19
        else:
            log_interval = 14

        loss, prev_loss = {}, {}
        total_acc, total_count, tot_loss = 0, 0, 0

        for idx, data in enumerate(zip(*dataloader.values())):
            self.running_iteration = idx + epoch * len(dataloader[0])
            vars, grads, m, v, v_hat, u_tilde, lr = {}, {}, {}, {}, {}, {}, {}

            for i in range(self.agents):
                self.agent_optimizers[i].zero_grad()
                inputs, labels = data[i]
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                predicted_label = self.agent_models[i](inputs)
                loss[i] = self.criterion(predicted_label, labels)
                loss[i].backward()
                vars[i], grads[i] = self.agent_optimizers[i].collect_params()

                if self.running_iteration == 0:
                    m[i] = []
                    v[i] = []
                    v_hat[i] = []
                    u_tilde[i] = []
                else:
                    m[i] = self.agent_optimizers[i].collect_m()
                    v[i] = self.agent_optimizers[i].collect_v()
                    v_hat[i] = self.agent_optimizers[i].collect_v_hat()
                    u_tilde[i] = self.agent_optimizers[i].collect_u_tilde()

                total_acc += (predicted_label.argmax(1) == labels).sum().item()
                total_count += labels.size(0)

                tot_loss += loss[i].item()

            for i in range(self.agents):
                self.agent_optimizers[i].step(self.running_iteration, vars=vars, m=m[i], v=v[i],
                                              v_hat=v_hat[i],
                                              u_tilde=u_tilde)
            if idx % log_interval == 0 and epoch % 1 == 0:
                self.it_logger(total_acc, total_count, epoch, log_interval, tot_loss, start_time)
                total_acc, total_count, tot_loss = 0, 0, 0
                for i in range(self.agents):
                    self.agent_models[i].train()
                start_time = perf_counter()
        return total_acc


