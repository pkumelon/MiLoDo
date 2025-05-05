import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import random as rm
import torch.nn.init as init
import os
import json

from algorithms import *
from MiLoDo import *
from optimizees import *
from utils import *

def set_seed(seed):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    rm.seed(seed)

def get_optimizees(args, device):
    """Create optimizee instances based on parameters"""
    if args.optimizee.lower() == 'lasso':
        return [CSOptimizee(args.agents, args.features, args.data, 
                           args.noise, device, l1_weight=args.l1_weight) 
               for _ in range(args.batch_size)]
    elif args.optimizee.lower() == 'logistic':
        return [L1LogisticRegression(args.agents, args.features, args.data, 
                                args.noise, device,l1_weight=args.l1_weight) 
              for _ in range(args.batch_size)]
    # Add other optimizee types
    
    elif args.optimizee.lower() == 'mlp':
        
        transform = transforms.Compose([
        transforms.ToTensor(),  
        transforms.Normalize((0.5,), (0.5,))  
    ])

 
        full_train_data = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
        test_data = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
        
        return [MLP_Optimizee(args.agents, device, full_train_data, args.subset_size, args.per_agent_batch_size) 
                for _ in range(args.batch_size)]
        
    elif args.optimizee == 'resnet':
        transform = transforms.Compose([
        transforms.Resize((32, 32)),  
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  
    ])

      
        full_trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=transform)
        full_testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=True, transform=transform)
        
        return [ResNet_Optimizee(args.agents, device, full_trainset,args.subset_size, args.per_agent_batch_size)
                for _ in range(args.batch_size)]
        
    else:
        raise ValueError(f"Unsupported optimizee type: {args.optimizee}")

def get_optimizer(optimizer_type, model_params, lr, weight_decay):
    """Create optimizer based on parameters"""
    if optimizer_type.lower() == 'adam':
        return torch.optim.Adam(model_params, lr=lr, weight_decay=weight_decay)
    elif optimizer_type.lower() == 'sgd':
        return torch.optim.SGD(model_params, lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

def main():
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create topology
    topo = generate_ring_topology(args.agents)
    
    # Initialize model
    model = MiLoDo_Optimizer(topo, 20, 1, args.use_bias, 
                             device, {'lr': args.lr_init})
    model.reset_state(args.batch_size)
    model.to(device)
    
    # Create optimizee problems
    optimizees = get_optimizees(args, device)
    
    # Multi-stage training
    print(f"Starting multi-stage training with {args.stages} stages")
    
    # Save training configuration
    training_info = {
        'args': vars(args),
        'stages': args.stages_data,
        'training_history': []
    }
    
    # Execute training stages
    for i, stage_config in enumerate(args.stages_data):
        lr = stage_config['lr']
        truncation_length = stage_config['truncation_length']
        iterations = stage_config['iterations']
        epochs = stage_config['epochs']
        
        print(f"\nStage {i+1}/{args.stages}:")
        print(f"  Learning rate: {lr}")
        print(f"  Truncation length (inner steps): {truncation_length}")
        print(f"  Iterations (outer steps): {iterations}")
        print(f"  Epochs: {epochs}")
        
        # Create optimizer for this stage
        optimizer = get_optimizer(args.optimizer, model.parameters(), lr, args.weight_decay)
        
        # Enable anomaly detection
        torch.autograd.set_detect_anomaly(True)
        
        # Train for this stage
        if args.optimizee.lower() == 'lasso' or args.optimizee == 'logistic':
            model = meta_train_l1(model, optimizees, optimizer, args.batch_size, 
                                        epochs, truncation_length, iterations, device)
        elif args.optimizee.lower() == 'mlp':
            MLP_now = MLP(device).to(device)
            model = meta_train_MLP(model,MLP_now, optimizees, optimizer, args.batch_size, 
                                        epochs, truncation_length, iterations, device)
            
        elif args.optimizee.lower() == 'resnet':
            resnet_now = ResNet(device).to(device)  
            model = meta_train_ResNet(model,resnet_now, optimizees, optimizer, args.batch_size, 
                                        epochs, truncation_length, iterations, device)
            
     
            
        
        # Record stage results
        stage_info = {
            'stage': i+1,
            'lr': lr,
            'truncation_length': truncation_length,
            'iterations': iterations,
            'epochs': epochs,
        }
        training_info['training_history'].append(stage_info)
        
        # Save intermediate model if requested
        if args.save_intermediate:
            intermediate_path = f"{os.path.splitext(args.save_path)[0]}_stage{i+1}.pth"
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'stage': i+1,
                'args': args,
                'training_info': training_info
            }, intermediate_path)
            print(f"  Intermediate model saved to {intermediate_path}")
    
    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'args': args,
        'training_info': training_info
    }, args.save_path)
    
    # Save training info as JSON
    training_info_path = f"{os.path.splitext(args.save_path)[0]}_training_info.json"
    with open(training_info_path, 'w') as f:
        # Convert non-serializable types to strings
        serializable_info = training_info.copy()
        serializable_info['args'] = {k: str(v) for k, v in serializable_info['args'].items()}
        json.dump(serializable_info, f, indent=4)
    
    print(f"\nTraining completed. Final model saved to {args.save_path}")
    print(f"Training information saved to {training_info_path}")

if __name__ == "__main__":
    main()