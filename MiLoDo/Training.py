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



def meta_train_l1(model, optimizees, optimizer, batch_size,  N_epochs,truncation_length, N_iterations,device):
    X_dim = optimizees[0].d
    N_batches = len(optimizees) // batch_size
    
    for epoch in range(N_epochs):
        sampling_index = np.random.permutation(len(optimizees))
        
        for batch in range(N_batches):
            model.reset_state(batch_size * X_dim)
            loss = 0
            X = torch.zeros(batch_size, model.n, X_dim).to(device)
            Y = torch.zeros(batch_size, model.n, X_dim).to(device)
            Z = torch.zeros(batch_size, model.n, X_dim).to(device)
            GX = torch.zeros(batch_size, model.n, X_dim).to(device)

            Lambda = torch.zeros(batch_size, X_dim)
            for i in range(batch_size):
                Lambda[i] = optimizees[sampling_index[i + batch * batch_size]].l1_weight
            
            for iteration in range(N_iterations):
                step_loss = 0
                for i in range(batch_size):
                    optimizee = optimizees[sampling_index[i + batch * batch_size]]
                    GX[i] = optimizee.gradient(X[i])
         
                X, Y, Z = model(X, Y, Z, GX, Lambda)
                
                for i in range(batch_size):
                    step_loss += optimizees[sampling_index[i + batch * batch_size]].loss(X[i])
                step_loss /= batch_size
                loss += step_loss 
                
                if (iteration + 1) % truncation_length == 0:
                    loss /= truncation_length
                    optimizer.zero_grad()
                    loss.backward()         
                    optimizer.step()
                    model.detach_state()
                    X = X.detach()
                    Y = Y.detach()
                    Z = Z.detach()
                    GX = GX.detach()

                    loss = 0
                    print(f'Epoch {epoch + 1}/{N_epochs}, Batch {batch + 1}/{N_batches}, Iter {iteration + 1}/{N_iterations}, Loss: {step_loss:.4f}')
                    
            print(f'Epoch {epoch + 1}/{N_epochs}, Batch {batch + 1}/{N_batches}, Final Loss: {step_loss:.4f}')
            print('='*30)
    return model





def meta_train_ResNet(model, resnet_now, optimizees, optimizer, batch_size, N_epochs, trunc_length, N_iterations, device):
    
    
    def ResNet_init(device):
        model = resnet_now
        init_vector = parameters_to_vector(model.parameters()).to(device).detach().clone()
        
        return init_vector
        
    
    X_dim = optimizees[0].d 
    N_agent = optimizees[0].n
    N_batches = len(optimizees)//batch_size 
    
    # model.register_full_backward_hook(hook_fn_backward)
    # model.register_forward_hook(hook_fn_forward)
    
    for epoch in range(N_epochs):
        print(f"Epoch {epoch + 1}/{N_epochs} started")
        sampling_index = np.random.permutation(len(optimizees))
        
        for batch in range(N_batches):
            model.reset_state(batch_size*X_dim)
            loss = 0 
            
            ##  Initialize X, Y, Z with same shapes. X.shape = (batch_size, N_agent, X_dim), where X[i][j] is a vector of length X_dim, which is the flattened parameter vector of MLP.
            X = torch.zeros(batch_size, N_agent, X_dim, requires_grad=True).to(device)
            for i in range(batch_size):
                for j in range(N_agent):
                    X[i][j] = ResNet_init(device).clone()
                    X[i][j].retain_grad()
                    
            Y = torch.zeros(batch_size, N_agent, X_dim, requires_grad=True).to(device)
            Z = torch.zeros(batch_size, N_agent, X_dim, requires_grad=True).to(device)
            GX = torch.zeros(batch_size, N_agent, X_dim, requires_grad=True).to(device)
   
           
            
            for iteration in range(N_iterations):
                step_loss = 0
                
                
                X.retain_grad()
                Y.retain_grad()
                Z.retain_grad()
                GX.retain_grad()
                
                ## Forward pass to get parameters
                
                G_next = torch.zeros(batch_size, N_agent, X_dim).to(device) 
                for i in range(batch_size):
                    optimizee = optimizees[sampling_index[batch*batch_size+i]]
                    G_next[i] = optimizee.gradient(X[i])
                GX = G_next.clone()
                GX.requires_grad = True
            
                
                X, Y, Z = model(X, Y, Z, GX)
                
                X.retain_grad()
                Y.retain_grad()
                Z.retain_grad()
                GX.retain_grad()
                
        
                
                ## Update main network parameters and compute loss
                for i in range(batch_size):
                    optimizee = optimizees[sampling_index[batch*batch_size+i]]
                    step_loss += optimizee.loss(X[i])
                    
                    
                step_loss = step_loss / batch_size
                loss = loss + step_loss

                
                
                ## Update model parameters
                if (iteration+1) % trunc_length == 0: 
                    loss = loss / trunc_length    
                    optimizer.zero_grad()
                    loss.backward()

                    optimizer.step()
                    model.detach_state()
                    X = X.detach()
                    Y = Y.detach()
                    Z = Z.detach()
                    GX = GX.detach()
                    
                    X.requires_grad = True
                    Y.requires_grad = True
                    Z.requires_grad = True
                    GX.requires_grad = True
                    
                    ## Reset gradients
                    optimizer.zero_grad()
                    model.zero_grad()
             
                    loss = 0
                    
                
                    print(f"Epoch {epoch + 1}/{N_epochs}, Batch {batch + 1}/{N_batches}, Iteration {iteration + 1}/{N_iterations}, Loss: {step_loss:.6f}")
 
      
                    
           
            print("-" * 30)
    return model
        
        


def meta_train_MLP(model, MLP_now, optimizees, optimizer, batch_size, N_epochs, trunc_length, N_iterations, device):
    
    def MLP_init(device):
        model = MLP_now
        init_vector = parameters_to_vector(model.parameters()).to(device).detach().clone()
        
        return init_vector
        
    
    X_dim = optimizees[0].d 
    N_agent = optimizees[0].n
    N_batches = len(optimizees)//batch_size 
    
    # model.register_full_backward_hook(hook_fn_backward)
    # model.register_forward_hook(hook_fn_forward)
    

    
    for epoch in range(N_epochs):
        print(f"Epoch {epoch + 1}/{N_epochs} started")
        sampling_index = np.random.permutation(len(optimizees))
        
        for batch in range(N_batches):
            model.reset_state(batch_size*X_dim)
            loss = 0 
            
            ## Initialize X, Y, Z with same shapes. X.shape = (batch_size, N_agent, X_dim), where X[i][j] is a vector of length X_dim, which is the flattened parameter vector of MLP.
            X = torch.zeros(batch_size, N_agent, X_dim, requires_grad=True).to(device)
            for i in range(batch_size):
                for j in range(N_agent):
                    X[i][j] = MLP_init(device).clone()
                    X[i][j].retain_grad()
                    
            Y = torch.zeros(batch_size, N_agent, X_dim, requires_grad=True).to(device)
            Z = torch.zeros(batch_size, N_agent, X_dim, requires_grad=True).to(device)
            GX = torch.zeros(batch_size, N_agent, X_dim, requires_grad=True).to(device)
   
           
            
            for iteration in range(N_iterations):
                step_loss = 0
                
                X.retain_grad()
                Y.retain_grad()
                Z.retain_grad()
                GX.retain_grad()
                
                G_next = torch.zeros(batch_size, N_agent, X_dim).to(device) 
                for i in range(batch_size):
                    optimizee = optimizees[sampling_index[batch*batch_size+i]]
                    G_next[i] = optimizee.gradient(X[i])
                GX = G_next.clone()
                GX.requires_grad = True
                
                ## Forward pass to get parameters
                
                X, Y, Z = model(X, Y, Z, GX)
                
                X.retain_grad()
                Y.retain_grad()
                Z.retain_grad()
                GX.retain_grad()
                
                
                ## Update main network parameters and compute loss
                for i in range(batch_size):
                    optimizee = optimizees[sampling_index[batch*batch_size+i]]
                    step_loss += optimizee.loss(X[i])
                    
                    
                step_loss = step_loss / batch_size
                loss = loss + step_loss

                # print(f"Epoch {epoch + 1}/{N_epochs}, Batch {batch + 1}/{N_batches}, Iteration {iteration + 1}/{N_iterations}, Loss: {step_loss:.6f}")
                
                ## Update model parameters
                if (iteration+1) % trunc_length == 0: 
                    loss = loss / trunc_length    
                    optimizer.zero_grad()
                    loss.backward()


                    optimizer.step()
                    model.detach_state()
                    X = X.detach()
                    Y = Y.detach()
                    Z = Z.detach()
                    GX = GX.detach()
                    
                    X.requires_grad = True
                    Y.requires_grad = True
                    Z.requires_grad = True
                    GX.requires_grad = True
                    
                    ## Reset gradients
                    optimizer.zero_grad()
                    model.zero_grad()
                
                    
                    loss = 0
                    
                    ## Output detailed epoch, iteration, loss information
                    print(f"Epoch {epoch + 1}, Batch {batch + 1}/{N_batches}, Iteration {iteration + 1}/{N_iterations}, Loss: {step_loss:.6f}")


           
            print("-" * 30)
    return model