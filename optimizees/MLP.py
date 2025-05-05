import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import random as rm
import torch.nn.init as init
import pickle
import torch
from torchvision import datasets, transforms
from torch.utils.data.dataset import Subset
from torch.utils.data import DataLoader
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.utils.data import ConcatDataset
from torch import nn,optim
import copy
import logging
from logging import FileHandler
from logging import StreamHandler
import pickle

class MLP(nn.Module):
    def __init__(self,device,hidden_size=16):  
        # Inherit from parent class
        super(MLP, self).__init__()
        # Create a three-layer network
        # Input 28*28 is the image size, output 10 is the number of digit classes
        hidden_first = hidden_size
        hidden_second = hidden_size
        self.first = nn.Linear(in_features=28*28, out_features=hidden_first).to(device)
        self.second = nn.Linear(in_features=hidden_first, out_features=hidden_second).to(device)
        self.third = nn.Linear(in_features=hidden_second, out_features=10).to(device)

    def forward(self, data):
        # First convert image data to 1*784 tensor
        data = data.view(-1, 28*28)
        data = F.relu(self.first(data))
        data = F.relu((self.second(data)))
        data =self.third(data)

        return data


## To create a distributed optimization MLP optimizee, we write an MLP_Optimizee class. Each node has its own MLP parameters, the dataset is distributed to different nodes, each node calculates its own gradient, and Loss takes the average. Specific parameter updates are handled by the MiLoDo algorithm.
class MLP_Optimizee(nn.Module):
    def __init__(self, N_agent, device, train_data, subset_size,batach_size,hidden_size=16,test_flag=False):
        super(MLP_Optimizee, self).__init__()
        self.n = N_agent
        self.device = device
        
        self.hidden_first = hidden_size
        self.hidden_second = hidden_size
        
        self.model = MLP(self.device)
        
        # Calculate the parameter size based on the defined MLP structure
        self.d = sum(p.numel() for p in self.model.parameters())

        # Randomly sample a subset of the dataset and then distribute to agents
        self.datasets = self.split_dataset_for_agents(train_data, N_agent, subset_size)
        
        self.batch_size = batach_size
    
        self.l1_weight = 0.0
        
        
        # Initialize DataLoader for each agent
        self.loaders = [
            iter(DataLoader(dataset, batch_size=self.batch_size, shuffle=True))
            for dataset in self.datasets
        ]
        
        self.data_now = [0]* self.n
        self.target_now = [0]*self.n
        
        self.test_flag = test_flag  
    
    ## Distribute data
    def split_dataset_for_agents(self, dataset, N_agent, subset_size):
        # Randomly sample a subset of the dataset
        indices = np.random.choice(len(dataset), subset_size, replace=False)
        sampled_dataset = Subset(dataset, indices)

        # Distribute the sampled dataset evenly among the agents
        num_items_per_agent = len(sampled_dataset) // N_agent
        datasets = []
        for i in range(N_agent):
            start_idx = i * num_items_per_agent
            end_idx = start_idx + num_items_per_agent if i < N_agent - 1 else len(sampled_dataset)
            subset = Subset(sampled_dataset, range(start_idx, end_idx))
            datasets.append(subset)
        return datasets
    
    def prox(self, matrix, input, batched=False):
        return torch.sign(input) * torch.max(torch.abs(input) - self.l1_weight * matrix, torch.zeros_like(input))


    def gradient(self,X):    ## Automatic gradient calculation
        if not self.test_flag:
            X.retain_grad()
        G = torch.zeros_like(X,device=self.device)
  
        # G.retain_grad()   
        for i in range(self.n):        
            
            # Use pre-created DataLoader
            try:
                data, target = next(self.loaders[i])
            except StopIteration:
                # If the current DataLoader is exhausted, reinitialize the iterator
                self.loaders[i] = iter(DataLoader(self.datasets[i], batch_size=self.batch_size, shuffle=True))
                data, target = next(self.loaders[i])
            
            ## Record this batch of data for calculating loss
            self.data_now[i]=data
            self.target_now[i]=target
            
            data, target = data.to(self.device), target.to(self.device)
            
            loss = 0
            temp_mlp = copy.deepcopy(self.model)
            # Convert parameter vector to model parameters
            
            num_el = 0
            for module in temp_mlp.children():
                for p_key in module._parameters:
                    p = module._parameters[p_key]
                    if p is not None:
                        numel = p.numel()
                        p = p - p + X[i][num_el:num_el+numel].reshape(p.shape)
                        module._parameters[p_key] = p
                        module._parameters[p_key].retain_grad()
                        num_el += numel
        #  module._parameters[p_key].retain_grad()
            output = temp_mlp(data)
            loss = F.cross_entropy(output, target)
            
            loss.backward(retain_graph=True) ## retain_graph brings gradient errors (1e-6 order of magnitude), but not using retain_graph will report second backward error.

            G[i] = parameters_to_vector( param.grad for param in temp_mlp.parameters())

       
        return G
            
                

    def loss (self,X):
        ## X is an MLP parameter matrix of shape (N_agent, self.d), each X[i] is a flattened parameter vector of an MLP
        # The function returns the average of all agents' losses, that is, each X[i] goes through forward calculation, then calculates its own loss. Finally, the average is returned.
        loss = 0
        for i in range(self.n):
            
           
            
            vector = X[i]
            # Create a new copy of the MLP model for each agent
            temp_mlp = copy.deepcopy(self.model)
            # Convert parameter vector to model parameters
            
            num_el = 0
            for module in temp_mlp.children():
                for p_key in module._parameters:
                    p = module._parameters[p_key]
                    if p is not None:
                        numel = p.numel()
                        p = p - p + vector[num_el:num_el+numel].reshape(p.shape)
                        module._parameters[p_key] = p
                        module._parameters[p_key].retain_grad()
                        num_el += numel
            
            data, target = self.data_now[i],self.target_now[i]   ## Only get the loss of the current batch, so you need to run the gradient function first, then the loss_compute function.
            data, target = data.to(self.device), target.to(self.device)
            output = temp_mlp(data)
            loss += F.cross_entropy(output, target)
            
        loss = loss/(self.n)
        
        return loss 
    


    
def MLP_init(MLP_now,device):
    model = MLP_now
    init_vector = parameters_to_vector(model.parameters()).to(device).detach().clone()
    
    return init_vector