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




class L1LogisticRegression():
    def __init__(self, N_agent, N_feature, N_data, noise_scale, device,l1_weight=0.1):
        self.device = device
        self.n = N_agent  # Number of agents
        self.d = N_feature  # Number of features
        self.m = N_data  # Number of data points
        self.k = N_data // N_agent  # Number of data points per agent
        self.sigma = noise_scale  # Noise scale
        self.A = torch.randn(self.m, self.d, device=self.device)  # Randomly generate data feature matrix

        # Create a sparse solution vector with a certain proportion of elements set to 0
        x = torch.randn(self.d, device=self.device)
        k = int(0.75 * self.d)  # Calculate the number of elements to set to 0
        non_zero_indices = torch.randperm(self.d)[:k] 
        x[non_zero_indices] = 0.0  # Set these elements to 0

        # Generate target variables for logistic regression, i.e., classification labels
        logits = self.A @ x
        self.b = (torch.sigmoid(logits) > 0.5).float()  # Use logistic function to generate 0/1 labels
        # self.b = (torch.sigmoid(logits) > 0.5).double()  # Use logistic function to generate 0/1 labels
        
        self.l1_weight = torch.tensor(l1_weight,device=self.device)

    def gradient(self, X):
        G = torch.zeros((self.n, self.d), device=self.device)  # Initialize gradient
        for i in range(self.n):
            Ai = self.A[self.k * i: self.k * (i + 1), :]  # Extract data features for each agent
            bi = self.b[self.k * i: self.k * (i + 1)].reshape(-1, 1)  # Extract labels for each agent
            preds = torch.sigmoid(X[i, :] @ Ai.T).reshape(-1,1)  # Calculate predictions
            G[i, :] = Ai.T @ (preds - bi).squeeze() / self.k  # Calculate gradient
        return G

    def loss(self, X, batched=False):
        if batched:
            x_bar = torch.mean(X, axis=0).reshape(X.shape[0], -1)  # Calculate batch mean
            preds = torch.sigmoid(x_bar @ self.A.T)  # Calculate batch predictions
            # Calculate batch binary cross-entropy loss and add L1 regularization term
            return F.binary_cross_entropy(preds, self.b.reshape(1, -1), reduction='mean') + self.l1_weight * torch.norm(x_bar, p=1, dim=1)
        x_bar = torch.mean(X, axis=0).reshape(1, -1)  # Calculate mean
        preds = torch.sigmoid(x_bar @ self.A.T)  # Calculate predictions
        # Calculate binary cross-entropy loss and add L1 regularization term
        return F.binary_cross_entropy(preds, self.b.reshape(1, -1), reduction='mean') + self.l1_weight * torch.norm(x_bar, p=1)

    def prox(self, matrix, input, batched=False):
        # Apply proximity operator for L1 regularization
        return torch.sign(input) * torch.max(torch.abs(input) - self.l1_weight * matrix, torch.zeros_like(input))
    
    
    ## The following is specifically for solving optimal loss ############################################
    def compute_gradient(self, X):
        """
        Calculate the gradient for logistic regression
        """
        preds = torch.sigmoid(self.A @ X)
        gradient = self.A.t() @ (preds - self.b) / self.m
        return gradient

    def proximal_operator(self, X, alpha):
        """
        Proximal operator for L1 regularization
        """
        return torch.sign(X) * torch.max(torch.abs(X) - alpha * self.l1_weight, torch.zeros_like(X))
    
    def loss_optimal(self, X):
        """
        Calculate logistic regression loss with L1 regularization term
        """
        preds = torch.sigmoid(self.A @ X)
        logistic_loss = F.binary_cross_entropy(preds, self.b, reduction='mean')
        l1_loss = self.l1_weight * torch.norm(X, p=1)
        return logistic_loss + l1_loss

    def proximal_gradient_descent(self, lr, n_iter):
        """
        Use proximal gradient descent algorithm for optimization
        """
        X = torch.zeros(self.d, device=self.device)  # Initialize parameters
        loss_history = []  # Initialize loss history record

        for i in range(n_iter):
            # Calculate gradient
            G = self.compute_gradient(X)
            # Gradient descent
            X = X - lr * G
            # Apply proximal operator
            X = self.proximal_operator(X, lr)
            # Calculate current loss and record
            current_loss = self.loss_optimal(X)
            loss_history.append(current_loss.item())
            # Optionally print loss at each iteration
            # print("Iteration:", i, "Loss:", current_loss.item())

        return X  # Return final weights and loss history
    ##########################################################################