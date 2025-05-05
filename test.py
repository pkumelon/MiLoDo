import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import random as rm
import torch.nn.init as init
import os
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets

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

def main():
    # =========== Test Parameter Settings ===========
    # Basic Settings
    seed = 42
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(seed)
    print(f"Using device: {device}")
    
    # Network Topology
    num_agents = 10
    topo = generate_ring_topology(num_agents)
    
    # Algorithm Parameters
    iterations = 100
    learning_rate = 0.01
    
    # Optimization Problem Type and Parameters (Choose one type)
    optimizee_type = 'lasso'  # Options: 'lasso', 'logistic', 'mlp', 'resnet'
    
    # LASSO/Logistic Parameters
    features = 300
    data_points = 100
    noise_level = 0.1
    l1_weight = 0.1
    
    # MLP/ResNet Parameters (Only used when these types are selected)
    subset_size = 1000
    per_agent_batch_size = 100
    
    # MiLoDo Model Settings
    use_bias = True
    # Choose whether to load pretrained model
    load_pretrained = False
    model_path = "./model/multi_stage_model.pth"  # Pretrained model path
    
    # Output Settings
    output_dir = "./fig"
    fig_name = f"{optimizee_type}-{num_agents}agents"
    
    # =========== Create Optimization Problem ===========
    mlp_now = None
    resnet_now = None
    
    if optimizee_type.lower() == 'lasso':
        
        optimizee = CSOptimizee(num_agents, features, data_points, noise_level, device, l1_weight=l1_weight)
    elif optimizee_type.lower() == 'logistic':
        optimizee = L1LogisticRegression(num_agents, features, data_points, noise_level, device, l1_weight=l1_weight)
    elif optimizee_type.lower() == 'mlp':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        full_train_data = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
        optimizee = MLP_Optimizee(num_agents, device, full_train_data, subset_size, per_agent_batch_size,test_flag=True)
        mlp_now =  MLP_init(MLP(device),device).to(device)
        
    elif optimizee_type.lower() == 'resnet':
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        full_trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        optimizee = ResNet_Optimizee(num_agents, device, full_trainset, subset_size, per_agent_batch_size,test_flag=True)
        resnet_now = ResNet_init(ResNet(device),device).to(device)  
    
    # =========== Create MiLoDo Model ===========
    milodo_model = MiLoDo_Optimizer(topo, 20, 1, use_bias, device, {'lr': learning_rate})
    
    if load_pretrained:
        print(f"Loading pretrained MiLoDo model from {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        milodo_model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print("Using randomly initialized MiLoDo model")
    
    # =========== Calculate Optimal Solution (if applicable) ===========
    if optimizee_type.lower() in ['lasso', 'logistic']:
        optimal_X = optimizee.proximal_gradient_descent(0.1, 5000)
        loss_optimal = optimizee.loss_optimal(optimal_X).item()
        print(f"Optimal Loss: {loss_optimal}")
        if hasattr(optimizee, 'A'):
            print(f"Condition number: {torch.linalg.cond(optimizee.A)}")
    else:
        # For neural network optimization, we don't have analytical optimal solution
        loss_optimal = 0
        print("No analytical optimal solution for neural network optimizations")
    
    # =========== Run Algorithms ===========
    print("\nRunning optimization algorithms...")
    
    # Algorithm 1: MiLoDo
    print("Running MiLoDo...")
    loss1 = MiLoDo(optimizee, iterations, device, milodo_model,mlp_now,resnet_now).run()
    
    # Algorithm 2: DGD
    print("Running Prox-DGD...")
    loss2 = DGD(optimizee, learning_rate, iterations, device, topo,mlp_now,resnet_now).run()
    
    # Algorithm 3: Prox-ED
    print("Running Prox-ED...")
    loss3 = Prox_ED(optimizee, learning_rate, iterations, topo,mlp_now,resnet_now).run()
    
    # Algorithm 4: Prox-ATC
    print("Running Prox-ATC...")
    loss4 = Prox_ATC(optimizee, learning_rate, iterations, topo,mlp_now,resnet_now).run()
    
    # Algorithm 5: PG-EXTRA
    print("Running PG-EXTRA...")
    loss5 = PG_EXTRA(optimizee, learning_rate, iterations, topo,mlp_now,resnet_now).run()
    
    # Algorithm 6: DAPG
    print("Running DAPG...")
    loss6 = DAPG(optimizee, learning_rate, iterations, topo,K=1,alpha=0.25,mlp_now=mlp_now,resnet_now=mlp_now).run()
    
    # Algorithm 7: ODAPG
    print("Running ODAPG...")
    loss7 = ODAPG(optimizee, learning_rate, iterations, topo,K=1, tau=0.2,mlp_now=mlp_now,resnet_now=mlp_now).run()
    
    # =========== Process Results ===========
    # If optimal solution exists, convert to absolute error
    if optimizee_type.lower() in ['lasso', 'logistic']:
        loss1 = [l - loss_optimal for l in loss1]
        loss2 = [l - loss_optimal for l in loss2]
        loss3 = [l - loss_optimal for l in loss3]
        loss4 = [l - loss_optimal for l in loss4]
        loss5 = [l - loss_optimal for l in loss5]
        loss6 = [l - loss_optimal for l in loss6]
        loss7 = [l - loss_optimal for l in loss7]
    
    # =========== Create Output Directory ===========
    os.makedirs(output_dir, exist_ok=True)
    
    # =========== Plot Results ===========
    plt.figure(figsize=(10, 6))
    
    # Plot each algorithm
    plt.semilogy(loss1, label='MiLoDo' + (' (Pretrained)' if load_pretrained else ''))
    plt.semilogy(loss2, label='Prox-DGD')
    plt.semilogy(loss3, label='Prox-ED')
    plt.semilogy(loss4, label='Prox-ATC')
    plt.semilogy(loss5, label='PG-EXTRA')
    plt.semilogy(loss6, label='DAPG')
    plt.semilogy(loss7, label='ODAPG')
    
    # Add chart details
    plt.xlabel('Iteration')
    plt.ylabel('f - f*' if optimizee_type.lower() in ['lasso', 'logistic'] else 'Loss')
    plt.title(f'Loss Comparison on {optimizee_type.upper()} Problem')
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.3)
    
    # Save chart
    fig_path = os.path.join(output_dir, f"{fig_name}.png")
    plt.savefig(fig_path, dpi=300)
    print(f"\nFigure saved to {fig_path}")

if __name__ == "__main__":
    main()