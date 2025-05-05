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



class ResNet(nn.Module):    
    
    def __init__(self,device, num_classes=10):
        super(ResNet, self).__init__()
        self.expansion =1 
        self.in_channels = 16
        self.device = device
        # Initial convolution layer and batch normalization
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False).to(device)
        self.bn1 = nn.BatchNorm2d(16).to(device)
        
        ## Residual block layer 1
        self.resblock1_conv1 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False).to(device)
        self.resblock1_bn1 = nn.BatchNorm2d(16).to(device)
        self.resblock1_conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False).to(device)
        self.resblock1_bn2 = nn.BatchNorm2d(16).to(device)
        self.resblock1_downsample = None
        
        ## Residual block layer 2
        self.resblock2_conv1 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False).to(device)
        self.resblock2_bn1 = nn.BatchNorm2d(32).to(device)
        self.resblock2_conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False).to(device)
        self.resblock2_bn2 = nn.BatchNorm2d(32).to(device)
        self.resblock2_downsample_1 = nn.Conv2d(16, 32, kernel_size=1, stride=2, bias=False).to(device)
        self.resblock2_downsample_2 = nn.BatchNorm2d(32).to(device)
        
        ## Residual block layer 3
        self.resblock3_conv1 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False).to(device)
        self.resblock3_bn1 = nn.BatchNorm2d(64).to(device)
        self.resblock3_conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False).to(device)
        self.resblock3_bn2 = nn.BatchNorm2d(64).to(device)
        self.resblock3_downsample_1 = nn.Conv2d(32, 64, kernel_size=1, stride=2, bias=False).to(device)
        self.resblock3_downsample_2 = nn.BatchNorm2d(64).to(device)
        
        # Average pooling and fully connected layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)).to(device)
        self.fc = nn.Linear(64 * self.expansion, num_classes).to(device)
        
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        
        ## Residual block layer 1
        identity = x
        x = self.resblock1_conv1(x)
        x = self.resblock1_bn1(x)
        x = F.relu(x)
        x = self.resblock1_conv2(x)
        x = self.resblock1_bn2(x)
        if self.resblock1_downsample is not None:
            identity = self.resblock1_downsample(identity)
        x += identity
        x = F.relu(x)
        
        ## Residual block layer 2
        identity = x
        x = self.resblock2_conv1(x)
        x = self.resblock2_bn1(x)
        x = F.relu(x)
        x = self.resblock2_conv2(x)
        x = self.resblock2_bn2(x)
        if self.resblock2_downsample_1 is not None:
            identity = self.resblock2_downsample_1(identity)
            identity = self.resblock2_downsample_2(identity)
        x += identity
        x = F.relu(x)
        
        ## Residual block layer 3
        identity = x
        x = self.resblock3_conv1(x)
        x = self.resblock3_bn1(x)
        x = F.relu(x)
        x = self.resblock3_conv2(x)
        x = self.resblock3_bn2(x)
        if self.resblock3_downsample_1 is not None:
            identity = self.resblock3_downsample_1(identity)
            identity = self.resblock3_downsample_2(identity)
        x += identity
        x = F.relu(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
                


class ResNet_Optimizee(nn.Module):
    def __init__(self, N_agent, device, train_data, subset_size,batch_size,test_flag=False):    
        super(ResNet_Optimizee, self).__init__()
        self.n = N_agent
        self.device = device
        
        # Define ResNet model
        self.model = ResNet(device, num_classes=10)
        
        # Calculate the parameter size of the ResNet model
        self.d = sum(p.numel() for p in self.model.parameters())
      
     
        
        # Randomly sample a subset of the dataset and distribute to agents
        self.datasets = self.split_dataset_for_agents(train_data, N_agent, subset_size)
        
        self.batch_size = batch_size
        
        # Initialize DataLoader for each agent
        self.loaders = [
            iter(DataLoader(dataset, batch_size=self.batch_size, shuffle=True))
            for dataset in self.datasets
        ]
        
        self.data_now = [0] * self.n
        self.target_now = [0] * self.n
        
        self.l1_weight = 0.0
        
        self.test_flag = test_flag
    
    def split_dataset_for_agents(self, dataset, N_agent, subset_size):
        # Randomly sample a subset of the dataset
        indices = np.random.choice(len(dataset), subset_size, replace=False)
        sampled_dataset = Subset(dataset, indices)

        # Distribute the sampled dataset evenly among agents
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
    
    def gradient(self,X):   
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
                self.loaders[i] = iter(DataLoader(self.datasets[i], batch_size=self.batch_size, shuffle=True)) ## Get current batch
                data, target = next(self.loaders[i])
            
            ## Record this batch of data for loss calculation
            self.data_now[i]=data
            self.target_now[i]=target
            
            data, target = data.to(self.device), target.to(self.device) 
            
            loss = 0
            temp_resnet = copy.deepcopy(self.model)
            # Convert parameter vector to model parameters
            
            num_el = 0
            for module in temp_resnet.children():
                for p_key in module._parameters:
                    p = module._parameters[p_key]
                    if p is not None:
                        numel = p.numel()
                        p = p - p + X[i][num_el:num_el+numel].reshape(p.shape)
                        module._parameters[p_key] = p
                        module._parameters[p_key].retain_grad()
                        num_el += numel
         
            output = temp_resnet(data)
            loss = F.cross_entropy(output, target)
            
            # ## Get corresponding data loader
            # loader = DataLoader(self.datasets[i], batch_size=self.batch_size, shuffle=True)
            # # For each agent, calculate the gradient
            # for data, target in loader:
            #     data, target = data.to(self.device), target.to(self.device)
            #     output = temp_resnet(data)
            #     loss += F.cross_entropy(output, target)
            
                
            loss.backward(retain_graph=True) ## retain_graph brings gradient errors (1e-6 order of magnitude), but not using retain_graph will report second backward error.
          

            G[i] = parameters_to_vector( param.grad for param in temp_resnet.parameters())

       
        return G
            

    def loss (self,X):
        ## X is a matrix of shape (N_agent, self.d) of MLP parameters, each X[i] is a flattened parameter vector
        # The function returns the average of all agents' losses, that is, each X[i] goes through forward calculation, then calculates its own loss. Finally, the average is returned.
        loss = 0
        for i in range(self.n):
            
            vector = X[i]
            # Create a new copy of the ResNet model for each agent
            temp_resnet = copy.deepcopy(self.model)
            # Convert parameter vector to model parameters
            
            num_el = 0
            for module in temp_resnet.children():
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
            output = temp_resnet(data)
            loss += F.cross_entropy(output, target)
            
            # # Get corresponding data loader  ## This traversal method is full traversal, not stochastic gradient descent
            # loader = DataLoader(self.datasets[i], batch_size=self.batch_size, shuffle=True)
            
            # # For each agent, calculate gradient
            # for data, target in loader:
            #     data, target = data.to(self.device), target.to(self.device)
            #     output = temp_resnet (data)
            #     loss += F.cross_entropy(output, target)
            
            
        
        # loss = loss/(self.n * len(loader))
        loss = loss/(self.n)
        
        # print("loss:",loss.item())  
        # exit()
        
        return loss 
    
def ResNet_init(resnet_now,device):
        model = resnet_now
        init_vector = parameters_to_vector(model.parameters()).to(device).detach().clone()
        
        return init_vector
    

