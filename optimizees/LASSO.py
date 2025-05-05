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



class CSOptimizee():
    def __init__(self, N_agent, N_feature, N_data, noise_scale,device,l1_weight=0.1):  # l1_weight≠0 is LASSO Regression. l1_weight=0 is Linear Regression
        self.device=device
        self.n = N_agent
        self.d = N_feature
        self.m = N_data
        self.k = N_data // N_agent # number of data per agent
        self.sigma = noise_scale
        self.A = torch.randn(self.m, self.d,device=self.device)
        
        # Create a sparse solution vector with approximately 75% elements set to 0
        x = torch.randn(self.d, device=self.device)
        k = int(0.75 * self.d)  # Calculate the number of elements to set to 0
        indices = torch.topk(torch.abs(x), k, largest=False).indices  # Find indices of the k elements with smallest absolute values
        x[indices] = 0.0  # Set these elements to 0
        
        self.b = self.A @ torch.randn(self.d,device=self.device) + self.sigma * torch.randn(self.m,device=self.device)
        
        # self.l1_weight = torch.randn(1,device)**2
        self.l1_weight = torch.tensor(l1_weight,device=self.device)
        
       
        

    def gradient(self, X):
        G = torch.zeros((self.n, self.d),device=self.device)
        for i in range(self.n):
            Ai = self.A[self.k * i: self.k * (i + 1), :]
            bi = self.b[self.k * i: self.k * (i + 1)].reshape(-1, 1)
            G[i, :] = (X[i, :] @ Ai.permute(1, 0) - bi.permute(1, 0)) @ Ai / self.k
        return G
    
    
    def loss(self, X, batched=False):
        if batched:
            x_bar = torch.mean(X, axis=1).reshape(X.shape[0], -1)
            return torch.norm(x_bar @ self.A.permute(1, 0) - self.b.reshape(1, -1), dim=1) ** 2 / (2 * self.m) + self.l1_weight * torch.norm(x_bar, p=1, dim=1)
        x_bar = torch.mean(X, axis=0).reshape(1, -1)
        return torch.norm(x_bar @ self.A.permute(1, 0) - self.b.reshape(1, -1)) ** 2 / (2 * self.m) + self.l1_weight * torch.norm(x_bar, p=1)
    
    def prox(self, matrix, input, batched=False):
        return torch.sign(input) * torch.max(torch.abs(input) - self.l1_weight * matrix, torch.zeros_like(input))
    
 
    
    
    ### The following is specifically for solving optimal loss ############################################
    def proximal_operator(self, X, alpha):
        # Proximal mapping for handling L1 regularization term
        return torch.sign(X) * torch.max(torch.abs(X) - alpha * self.l1_weight, torch.zeros_like(X))

    def compute_gradient(self, X):
        # Calculate gradient
        G = (X @ self.A.T - self.b) @ self.A / self.m
        return G
    
    def loss_optimal(self, X):
        # Calculate the loss value for LASSO problem
        return torch.norm(X @ self.A.T - self.b) ** 2 / (2 * self.m) + self.l1_weight * torch.norm(X, p=1)
    
    

    def proximal_gradient_descent(self, lr, n_iter):
        
        # Use proximal gradient descent algorithm to find the optimal solution
        X = torch.zeros(self.d, device=self.device)  # Initialize parameters
        loss_history = []  # Initialize loss list

        for i in range(n_iter):
            # Calculate gradient
            G = self.compute_gradient(X)
            # Simple gradient descent step
            X = X - lr * G
            # Apply proximal mapping
            X = self.proximal_operator(X, lr)
            # Calculate current loss and record
            current_loss = self.loss_optimal(X)
    
            # print("Iteration:", i, "Loss:", current_loss)


    
        return X # Return final weights
    ######################################################################