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
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
from PIL import Image

import resnet
import torch.utils.data as data
import threading
from models import *
from ops import Algorithm2, DSGDN, DADAM, DAMSGrad, ATCDIGing, DSGD
import swanlab
import torch.optim as optim

warnings.filterwarnings("ignore")


class DTrainer:
    def __init__(self,
                 dataset="cifar10",
                 epochs=20,
                 batch_size=128,
                 lr=0.02,
                 agents=5,
                 w=None,
                 fname=None,
                 stratified=True):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_iterations = []
        self.train_accuracy = []
        self.test_accuracy_top1 = []
        self.test_accuracy_top5 = []
        self.train_loss = []
        self.test_loss = []
        self.lr_logs = []

        self.dataset = dataset
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.agents = agents
        self.running_iteration = 0
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
            file.writerow([f"{self.opt_name}, {self.batch_size}, {self.epochs}"])
            file.writerow(self.train_iterations)
            file.writerow(["train_accuracy"])
            file.writerow(self.train_accuracy)
            file.writerow(["test_accuracy_top1"])
            file.writerow(self.test_accuracy_top1)
            file.writerow(["test_accuracy_top5"])
            file.writerow(self.test_accuracy_top5)
            file.writerow(["train_loss"])
            file.writerow(self.train_loss)
            file.writerow(["test_loss"])
            file.writerow(self.test_loss)
            file.writerow(["ETA"])
            file.writerow(self.lr_logs)
            file.writerow([])
            file.writerow([])

    def homogeneous_distribution(self, trainset, testset):
        dataset_size = len(trainset)
        # 计算每个子集的大小
        split_sizes = [dataset_size // self.agents] * self.agents
        remainder = dataset_size % self.agents
        for i in range(remainder):
            split_sizes[i] += 1
        # 随机抽样索引
        indices = np.random.permutation(dataset_size)
        # 创建子集
        start_idx = 0
        for i in range(self.agents):
            subset_indices = indices[start_idx:start_idx + split_sizes[i]]
            subset_sampler = data.SubsetRandomSampler(subset_indices)
            self.train_loader[i] = data.DataLoader(trainset, batch_size=self.batch_size, sampler=subset_sampler,
                                                   pin_memory=True, num_workers=2)
            start_idx += split_sizes[i]

        self.test_loader = torch.utils.data.DataLoader(testset, batch_size=self.batch_size,
                                                   pin_memory=True, num_workers=3)

    def heterogeneous_distribution(self, trainset, testset):
        targets = np.array([trainset[i][1] for i in range(len(trainset))])
        indices_per_class = {i: np.where(targets == i)[0] for i in range(self.class_num)}
        primary_percentage = 0.5

        agent_indices = [[] for _ in range(self.agents)]

        # 计算每个智能体负责的块大小
        block_size = self.class_num // self.agents

        for agent in range(self.agents):
            primary_classes = list(range(agent * block_size, (agent + 1) * block_size))
            secondary_classes = list(set(range(self.class_num)) - set(primary_classes))

            for cls in primary_classes:
                np.random.shuffle(indices_per_class[cls])
                primary_count = int(primary_percentage * len(indices_per_class[cls]))
                agent_indices[agent].extend(indices_per_class[cls][:primary_count])

            for cls in secondary_classes:
                np.random.shuffle(indices_per_class[cls])
                secondary_count = int((1 - primary_percentage) * len(indices_per_class[cls]))
                agent_indices[agent].extend(indices_per_class[cls][:secondary_count])

        # 创建 DataLoader
        for i in range(self.agents):
            temp_train = torch.utils.data.Subset(trainset, agent_indices[i])
            temp_train_size = len(temp_train)
            print(f'agent_train_size: {temp_train_size}')
            self.train_loader[i] = torch.utils.data.DataLoader(temp_train, batch_size=self.batch_size, shuffle=True)

        self.test_loader = torch.utils.data.DataLoader(testset, batch_size=self.batch_size,
                                                       pin_memory=True, num_workers=3)


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

        elif self.dataset == "imagenet":
            transform_train = transforms.Compose([
                transforms.Resize((336, 336)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            transform_test = transforms.Compose([
                transforms.Resize((336, 336)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            self.class_num = 1000
            trainset = datasets.ImageFolder('./data/imagenet/train', transform_train)
            testset = datasets.ImageFolder('./data/imagenet/sort_val', transform_test)

        if self.stratified:
            self.homogeneous_distribution(trainset, testset)
        else:
            self.heterogeneous_distribution(trainset, testset)

    def agent_setup(self):
        self.agent_models = {}
        self.test_agent_models = {}
        self.model_sgd = None
        self.test_model_sgd = None
        self.prev_agent_models = {}
        self.agent_optimizers = {}
        self.prev_agent_optimizers = {}
        model = None
        model1 = None

        if self.dataset == 'cifar10':
            model = CIFAR10CNN()
            model1= CIFAR10CNN()
        elif self.dataset == "imagenet":
            model = resnet.resnet1(self.class_num)
            model1=resnet.resnet1(self.class_num)

        for i in range(self.agents):
            if i == 0:
                if int(torch.cuda.device_count()) > 1:
                    self.agent_models[i] = model
                    self.test_agent_models[i] = model1
                else:
                    self.agent_models[i] = model
                    self.test_agent_models[i] = model1

            else:
                if int(torch.cuda.device_count()) > 1:
                    self.agent_models[i] = copy.deepcopy(self.agent_models[0])
                    self.test_agent_models[i] = copy.deepcopy(self.test_agent_models[0])
                else:
                    self.agent_models[i] = copy.deepcopy(model)
                    self.test_agent_models[i] = copy.deepcopy(self.test_agent_models[0])


            self.agent_models[i].to(self.device)
            self.agent_models[i].train()

            if self.opt_name == "Algorithm2":
                self.prev_agent_models[i] = copy.deepcopy(model)
                self.prev_agent_models[i].to(self.device)
                self.prev_agent_models[i].train()
                self.prev_agent_optimizers[i] = self.opt(
                    params=self.prev_agent_models[i].parameters(),
                    idx=i,
                    w=self.w,
                    agents=self.agents,
                    lr=self.lr,
                    name=self.opt_name,
                    device=self.device,
                    stratified=self.stratified
                )

            self.agent_optimizers[i] = self.opt(
                params=self.agent_models[i].parameters(),
                idx=i,
                w=self.w,
                agents=self.agents,
                lr=self.lr,
                name=self.opt_name,
                device=self.device,
                stratified=self.stratified
            )

    def eval(self, dataloader, agent_models, running_iteration):
        total_top1_acc, total_top5_acc, total_count = 0, 0, 0
        tot_t_loss = 0
        device = torch.device("cuda:1" if torch.cuda.device_count() > 1 else "cuda:0")
        for i in range(self.agents):
            agent_models[i].eval()
            agent_models[i].to(device)
        with torch.no_grad():
            # idx=0
            # s1=time.time()
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                for i in range(self.agents):
                    predicted_label = agent_models[i](inputs)
                    tot_t_loss += self.criterion(predicted_label, labels).item()

                    total_top1_acc += (predicted_label.argmax(1) == labels).sum().item()

                    _, top5_pred = predicted_label.topk(5, dim=1)
                    correct_top5 = top5_pred.eq(labels.view(-1, 1).expand_as(top5_pred))
                    total_top5_acc += correct_top5.sum().item()

                    total_count += labels.size(0)


        avg_top1_acc = total_top1_acc / total_count
        avg_top5_acc = total_top5_acc / total_count
        self.test_accuracy_top1.append(avg_top1_acc)
        self.test_accuracy_top5.append(avg_top5_acc)

        return avg_top1_acc, avg_top5_acc, tot_t_loss

    def it_logger(self, total_acc, total_count, epoch, log_interval, tot_loss, start_time, agent_optimizers,
                  running_iteration, agent_models):
        time1 = perf_counter() - start_time
        self._log(total_acc / total_count)
        # s1 = time.time()
        t1_acc, t5_acc, tot_t_loss = self.eval(self.test_loader, agent_models, running_iteration)

        self.train_loss.append(tot_loss / (self.agents * log_interval))
        self.test_loss.append(tot_t_loss / (self.agents * log_interval))

        total_sum_step = 0
        for i in range(self.agents):
            total_sum_step += agent_optimizers[i].collect_lr()
        average_step = total_sum_step / self.agents
        self.lr_logs.append(average_step)

        print(
            f"Epoch: {epoch + 1}, Iteration: {running_iteration}, " +
            f"Accuracy: {total_acc / total_count:.4f}, " +
            f"Test Accuracy top1: {t1_acc:.4f}, " +
            f"Test Accuracy top5: {t5_acc:.4f}, " +
            f"Train Loss: {tot_loss / (self.agents * log_interval):.4f}, " +
            f"Test Loss: {tot_t_loss / (self.agents * log_interval):.4f}, " +
            f"stepsize: {average_step:.4f}, " +
            f"Time taken: {time1:.4f}"
        )

    def trainer(self):
        print(
            f"==> Starting Training for {self.opt_name}, {self.epochs} epochs and {self.agents} agents on the {self.dataset} dataset, via {self.device}")
        for i in range(self.epochs):
            self.epoch_iterations(i)


class Algorithm2Trainer(DTrainer):
    def __init__(self, *args, **kwargs):
        self.opt = Algorithm2
        self.opt_name = "Algorithm2"
        super().__init__(*args, **kwargs)

        self.trainer()
        self._save()

    def epoch_iterations(self, epoch):
        start_time = perf_counter()

        log_interval = 14

        loss, prev_loss = {}, {}
        total_acc, total_count, tot_loss = 0, 0, 0

        for idx, data in enumerate(zip(*self.train_loader.values())):
            self.running_iteration = idx + epoch * len(self.train_loader[0])
            vars, grads, s, prev_grads, prev_vars, lr, prev_lr, b = {}, {}, {}, {}, {}, {}, {}, {}

            seed = 30
            random.seed(seed)
            os.environ['PYTHONHASHSEED'] = str(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            # some cudnn methods can be random even after fixing the seed
            # unless you tell it to be deterministic
            torch.backends.cudnn.deterministic = True

            # average_grads_dict = {}
            # average_prev_grads_dict = {}
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
                    lr[i] = self.lr
                    prev_lr[i] = self.lr
                    prev_grads[i] = 0
                else:
                    s[i] = self.agent_optimizers[i].collect_s()
                    lr[i] = self.agent_optimizers[i].collect_lr()
                    prev_lr[i] = self.agent_optimizers[i].collect_prev_lr()
                    prev_grads[i] = self.agent_optimizers[i].collect_prev_grads()

                self.prev_agent_models[i].load_state_dict(self.agent_models[i].state_dict())

                total_acc += (predicted_label.argmax(1) == labels).sum().item()
                total_count += labels.size(0)

                tot_loss += loss[i].item()

            for i in range(self.agents):
                self.agent_optimizers[i].step(self.running_iteration, vars=vars, grads=grads, s_all=s,
                                                       prev_vars=prev_vars, prev_grads=prev_grads,
                                                       lr_all=lr, lr_prev=prev_lr)

            if 0 == idx % log_interval and idx > 0:
                for i in range(self.agents):
                    self.test_agent_models[i].load_state_dict(self.agent_models[i].state_dict())
                logthread = threading.Thread(target=self.it_logger, args=((
                    total_acc, total_count, epoch, log_interval, tot_loss, start_time, self.agent_optimizers,
                    self.running_iteration, self.test_agent_models)))
                logthread.start()
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

    def epoch_iterations(self, epoch):
        start_time = perf_counter()

        log_interval = 14

        loss, prev_loss = {}, {}
        total_acc, total_count, tot_loss = 0, 0, 0

        for idx, data in enumerate(zip(*self.train_loader.values())):
            self.running_iteration = idx + epoch * len(self.train_loader[0])
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

            if idx % log_interval == 0 and idx > 0:
                for i in range(self.agents):
                    self.test_agent_models[i].load_state_dict(self.agent_models[i].state_dict())
                logthread = threading.Thread(target=self.it_logger, args=((
                    total_acc, total_count, epoch, log_interval, tot_loss, start_time, self.agent_optimizers,
                    self.running_iteration, self.test_agent_models)))
                logthread.start()

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

    def epoch_iterations(self, epoch):
        start_time = perf_counter()

        log_interval = 14

        loss, prev_loss = {}, {}
        total_acc, total_count, tot_loss = 0, 0, 0

        for idx, data in enumerate(zip(*self.train_loader.values())):
            self.running_iteration = idx + epoch * len(self.train_loader[0])
            vars, grads, m, v, v_hat, lr = {}, {}, {}, {}, {}, {}

            seed = 30
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
            if 0 == idx % log_interval and idx > 0:

                for i in range(self.agents):
                    self.test_agent_models[i].load_state_dict(self.agent_models[i].state_dict())
                logthread = threading.Thread(target=self.it_logger, args=((
                    total_acc, total_count, epoch, log_interval, tot_loss, start_time, self.agent_optimizers,
                    self.running_iteration, self.test_agent_models)))
                logthread.start()
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

    def epoch_iterations(self, epoch, ):
        start_time = perf_counter()

        log_interval = 14

        loss, prev_loss = {}, {}
        total_acc, total_count, tot_loss = 0, 0, 0

        for idx, data in enumerate(zip(*self.train_loader.values())):
            self.running_iteration = idx + epoch * len(self.train_loader[0])
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
            if idx % log_interval == 0 and idx>0:
                for i in range(self.agents):
                    self.test_agent_models[i].load_state_dict(self.agent_models[i].state_dict())
                logthread = threading.Thread(target=self.it_logger, args=((
                    total_acc, total_count, epoch, log_interval, tot_loss, start_time, self.agent_optimizers,
                    self.running_iteration, self.test_agent_models)))
                logthread.start()
                total_acc, total_count, tot_loss = 0, 0, 0
                for i in range(self.agents):
                    self.agent_models[i].train()
                start_time = perf_counter()
        return total_acc

class ATCDIGingTrainer(DTrainer):
    def __init__(self, *args, **kwargs):
        self.opt = ATCDIGing
        self.opt_name = "ATCDIGing"
        super().__init__(*args, **kwargs)
        self.trainer()
        self._save()

    def epoch_iterations(self, epoch):
        start_time = perf_counter()
        log_interval = 14

        loss, prev_loss = {}, {}
        total_acc, total_count, tot_loss = 0, 0, 0

        for idx, data in enumerate(zip(*self.train_loader.values())):
            self.running_iteration = idx + epoch * len(self.train_loader[0])
            vars, grads, s, prev_grads, lr = {}, {}, {}, {}, {}

            for i in range(self.agents):
                self.agent_optimizers[i].zero_grad()
                inputs, labels = data[i]
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                predicted_label = self.agent_models[i](inputs)
                loss[i] = self.criterion(predicted_label, labels)
                loss[i].backward()
                vars[i], grads[i] = self.agent_optimizers[i].collect_params()

                if self.running_iteration == 0:
                    s[i] = grads[i]
                    lr[i] = self.lr
                    prev_grads[i] = 0
                else:
                    s[i] = self.agent_optimizers[i].collect_s()
                    lr[i] = self.agent_optimizers[i].collect_lr()
                    prev_grads[i] = self.agent_optimizers[i].collect_prev_grads()

                total_acc += (predicted_label.argmax(1) == labels).sum().item()
                total_count += labels.size(0)

                tot_loss += loss[i].item()

            for i in range(self.agents):
                self.agent_optimizers[i].step(self.running_iteration, vars=vars, grads=grads, s_all=s, prev_grads=prev_grads,
                                                          lr_all=lr)
            if idx % log_interval == 0 and idx > 0:
                for i in range(self.agents):
                    self.test_agent_models[i].load_state_dict(self.agent_models[i].state_dict())
                logthread = threading.Thread(target=self.it_logger, args=((
                    total_acc, total_count, epoch, log_interval, tot_loss, start_time, self.agent_optimizers,
                    self.running_iteration, self.test_agent_models)))
                logthread.start()
                total_acc, total_count, tot_loss = 0, 0, 0
                for i in range(self.agents):
                    self.agent_models[i].train()
                start_time = perf_counter()
        return total_acc


class DSGDTrainer(DTrainer):
    def __init__(self, *args, **kwargs):
        self.opt = DSGD
        self.opt_name = "DSGD"
        super().__init__(*args, **kwargs)
        self.trainer()
        self._save()

    def epoch_iterations(self, epoch):
        start_time = perf_counter()
        log_interval = 14

        loss, prev_loss = {}, {}
        total_acc, total_count, tot_loss = 0, 0, 0

        for idx, data in enumerate(zip(*self.train_loader.values())):
            self.running_iteration = idx + epoch * len(self.train_loader[0])
            vars, grads, lr = {}, {}, {}

            for i in range(self.agents):
                self.agent_optimizers[i].zero_grad()
                inputs, labels = data[i]
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                predicted_label = self.agent_models[i](inputs)
                loss[i] = self.criterion(predicted_label, labels)
                loss[i].backward()
                vars[i], grads[i] = self.agent_optimizers[i].collect_params()

                total_acc += (predicted_label.argmax(1) == labels).sum().item()
                total_count += labels.size(0)

                tot_loss += loss[i].item()

            for i in range(self.agents):
                self.agent_optimizers[i].step(self.running_iteration, vars=vars, grads=grads)

            if idx % log_interval == 0 and idx > 0:
                for i in range(self.agents):
                    self.test_agent_models[i].load_state_dict(self.agent_models[i].state_dict())
                logthread = threading.Thread(target=self.it_logger, args=((
                    total_acc, total_count, epoch, log_interval, tot_loss, start_time, self.agent_optimizers,
                    self.running_iteration, self.test_agent_models)))
                logthread.start()
                total_acc, total_count, tot_loss = 0, 0, 0
                for i in range(self.agents):
                    self.agent_models[i].train()
                start_time = perf_counter()
        return total_acc