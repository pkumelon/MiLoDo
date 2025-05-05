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
import logging
from torch import nn,optim
import copy

# # Implementation of MiLoDo algorithm
class LSTM_Module(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, N_layer, use_bias,device, final_activation_type=None):
        super().__init__()
   
        self.device=device
        self.state = None
        self.layers = N_layer
        self.lstm = nn.LSTM(input_size, hidden_size, N_layer, bias=use_bias)
        self.linear = nn.Linear(hidden_size, hidden_size, bias=use_bias)
        self.linear_out = nn.Linear(hidden_size, output_size, bias=use_bias)
        self.final_activation = final_activation_type
        
   
    ## Randomly initialize hidden state  
    def reset_state(self, batch_size):  
        self.state = (torch.randn(self.layers, batch_size, self.lstm.hidden_size, device=self.device),
                      torch.randn(self.layers, batch_size, self.lstm.hidden_size, device=self.device))
        
    def detach_state(self):
        self.state = (self.state[0].detach(), self.state[1].detach())
    def forward(self, input):
        if self.state is None:
            self.reset_state(input.shape[0])
        x, self.state = self.lstm(input.unsqueeze(0), self.state)
        x = self.linear(x.squeeze(0))
        x = F.relu(x)
        x = self.linear_out(x)
        if self.final_activation == 'relu':
            x = F.relu(x)
        elif self.final_activation == 'softmax':
            x = F.softmax(x, dim=1)
        elif self.final_activation == 'exponential':
            x = torch.exp(x)
        return x
   
   
class M_Module(LSTM_Module):
    def __init__(self, N_neighbor, hidden_size, N_layer, use_bias,device, special_init=None): # N_neighbor= N_degree - 1
        super().__init__( 2, hidden_size, 1, N_layer, use_bias,device, 'relu')
        if special_init is not None:

            if self.final_activation == 'relu':
                lr = special_init['lr']

                # generate initial bias
                bias_init = torch.zeros(1)
                bias_init[-1] = lr
               

                # set initial bias
                self.linear_out.bias.data = bias_init

                # set initial weight
                noise_level = 0.001
                init.normal_(self.linear_out.weight, mean=0.0, std=noise_level)


class S_Module(LSTM_Module):
    def __init__(self, N_neighbor, hidden_size, N_layer, use_bias, device,special_init=None): # N_neighbor= N_degree - 1
        super().__init__(N_neighbor, hidden_size, N_neighbor, N_layer,use_bias, device, 'exponential')
        if special_init is not None:
            
            if self.final_activation == 'exponential':

                lr = special_init['lr']

                # generate initial bias
                bias_init = torch.zeros(N_neighbor)
                bias_init[:] = 1 / (2 * (N_neighbor + 1) * lr)
                bias_init = torch.log(bias_init)

                # set initial bias
                self.linear_out.bias.data = bias_init

                # set initial weight
                noise_level = 0.001
                init.normal_(self.linear_out.weight, mean=0.0, std=noise_level)

class C_Module(LSTM_Module):
       def __init__(self, N_neighbor, hidden_size, N_layer, use_bias, device,special_init=None): # N_neighbor= N_degree - 1
        super().__init__(N_neighbor, hidden_size, N_neighbor, N_layer, use_bias, device, 'relu')
        if special_init is not None:
            
            if self.final_activation == 'relu':

                lr = special_init['lr']

                # generate initial bias
                bias_init = torch.zeros(N_neighbor)
                bias_init[:] = 1 / (2 * (N_neighbor + 1))
           

                # set initial bias
                self.linear_out.bias.data = bias_init

                # set initial weight
                noise_level = 0.001
                init.normal_(self.linear_out.weight, mean=0.0, std=noise_level)
                
                
class MiLoDo_Optimizer(nn.Module):
    def __init__(self, graph_topology, hidden_size, N_layer, use_bias, device,special_init=None):
        super().__init__()
        self.device = device
        self.n = graph_topology.shape[0] 
        self.neighbors = {i:[] for i in range(self.n)} 
        for i in range(self.n):
            for j in range(self.n):
                if i != j and graph_topology[i, j] == 1: 
                    self.neighbors[i].append(j) 
        self.M = nn.ModuleList([M_Module(len(self.neighbors[i]), hidden_size, N_layer, use_bias, device, special_init) for i in range(self.n)]).to(device)
        self.S = nn.ModuleList([S_Module(len(self.neighbors[i]), hidden_size, N_layer, use_bias, device, special_init) for i in range(self.n)]).to(device)
        self.C = nn.ModuleList([C_Module(len(self.neighbors[i]), hidden_size, N_layer, use_bias, device, special_init) for i in range(self.n)]).to(device)
        self.max_degree = max([len(self.neighbors[i]) for i in range(self.n)])
        self.index_matrix = self._generate_index_tensor()

    def reset_state(self, batch_size): 
        for i in range(self.n):
            self.M[i].reset_state(batch_size) # Reset M module state 
            self.S[i].reset_state(batch_size) # Reset S module state
            self.C[i].reset_state(batch_size)

    def detach_state(self):
        for i in range(self.n):
            self.M[i].detach_state() # Detach M module state
            self.S[i].detach_state() # Detach S module state    
            self.C[i].detach_state()
            
    def forward(self, X, Y,Z, GX,Lambda= torch.tensor(0.0)):

        X, Y, Z,GX = X.to(self.device), Y.to(self.device), Z.to(self.device), GX.to(self.device)
        Y_inputs = Y.permute(0, 2, 1).reshape(-1, Y.shape[1]) 
 
        GX_inputs = GX.permute(0, 2, 1).reshape(-1, GX.shape[1]) 
        X_inputs = [0] * self.n 
        M_outputs = [0] * self.n 
        Z_next = torch.zeros(Z.shape, device=self.device) 
        
        ## Calculate Z_next
        for i in range(self.n):
            X_inputs[i] = torch.cat([Y_inputs[:, i].reshape(-1, 1), GX_inputs[:, i].reshape(-1, 1)], dim=1) 
            M_outputs[i] = self.M[i](X_inputs[i])
            Z_next_mid = X[:, i, :]  - M_outputs[i].reshape(X.shape[0], X.shape[2]) * (GX[:, i, :] + Y[:, i, :]) 
            Z_next[: ,i, :] = self._prox( M_outputs[i].reshape(X.shape[0], X.shape[2]),Z_next_mid,Lambda)
           
        ## Calculate Y_next    
        Z_next_differences = [torch.zeros(Z.shape[0], len(self.neighbors[i]), Z.shape[2],device=self.device) for i in range(self.n)]
        S_outputs = torch.zeros((self.n, Y.shape[0] * Y.shape[2], self.max_degree),device=self.device)     # (n, batch_size * feature_size, max_degree)
        Y_next = torch.zeros(Y.shape,device=self.device)
        for i in range(self.n):
            for e, j in enumerate(self.neighbors[i]):
                Z_next_differences[i][:, e, :] = Z_next[:, i, :] - Z_next[:, j, :]
            Z_next_differences[i] = Z_next_differences[i].permute(0, 2, 1)     # (batch_size, feature_size, nodes)
            Z_next_differences[i] = Z_next_differences[i].reshape(-1, Z_next_differences[i].shape[2])     # (batch_size * feature_size, degree)
            
            inputs2 = torch.cat((Z_next_differences[i],
                        ), dim=1)
            
            S_outputs[i, :, :len(self.neighbors[i])] = self.S[i](inputs2).reshape(Y.shape[0] * Y.shape[2], -1)
        
        for i in range(self.n):
            Y_next[:, i, :] = Y[:, i, :] + (((S_outputs[i, :, :len(self.neighbors[i])] + S_outputs[self.neighbors[i], :, self.index_matrix[i]].permute(1, 0)) / 2) * Z_next_differences[i]).sum(dim=1).reshape(Y.shape[0], Y.shape[2])
           
            
        ## Calculate X_next
        C_outputs = [0] * self.n 
        X_next = torch.zeros(X.shape,device=self.device)
        for i in range(self.n):
            inputs3 = torch.cat((Z_next_differences[i],
                        ), dim=1)
            C_outputs[i] = self.C[i](inputs3)
            
       
            
            X_next[:, i, :] = Z_next[:, i, :] - (C_outputs[i]* Z_next_differences[i]).sum(dim=1).reshape(Z.shape[0], Z.shape[2])

        # X_next.retain_grad()
        # Y_next.retain_grad()
        # Z_next.retain_grad()
            
        return X_next, Y_next, Z_next
    
    
    
    def _generate_index_tensor(self):
        res = {i:[] for i in range(self.n)}
        for i in range(self.n):
            for e, j in enumerate(self.neighbors[i]):
                res[j].append(e)
        return res 
    
    def _prox(self, matrix, inputs, Lambda):
        matrix = matrix.to(self.device)
        
        # Handle cases where Lambda is either a float or a tensor
        if isinstance(Lambda, (int, float)):
            Lambda_tensor = torch.tensor(Lambda, device=self.device)
        else:
            Lambda_tensor = Lambda.to(self.device)
        
        # Calculate soft-thresholding operation
        return torch.sign(inputs) * torch.max(
            torch.abs(inputs) - Lambda_tensor * matrix, 
            torch.zeros_like(inputs)
        )