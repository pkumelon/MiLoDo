import argparse
import json

def get_parser():
    """
    Create and return command line argument parser
    """
    parser = argparse.ArgumentParser(description='Training script for MiLoDo')
    
    # Model parameters
    parser.add_argument('--use_bias', action='store_true', help='Use bias in layers')
    parser.add_argument('--lr_init', type=float, default=0.03, help='special initialization for model')
    
    # Optimizer base parameters
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'], help='Optimizer type')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay for optimizer')
    
    # Multi-stage training parameters
    parser.add_argument('--stages', type=int, default=1, help='Number of training stages')
    parser.add_argument('--stages_config', type=str, default=None, 
                        help='Path to JSON file containing multi-stage training configuration')
    
    # Basic training parameters (used if stages_config is not provided)
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--agents', type=int, default=10, help='Number of agents (nodes)')
    parser.add_argument('--lr', type=float, default=0.0005, help='Learning rate for optimizer (single stage)')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs (single stage)')
    parser.add_argument('--truncation_length', type=int, default=5, 
                        help='Truncation length for inner loop (single stage)')
    parser.add_argument('--iterations', type=int, default=10, 
                        help='Iterations for outer loop (single stage)')
    
    # Optimizee parameters
    parser.add_argument('--optimizee', type=str, default='lasso', choices=['lasso', 'logistic', 'mlp', 'resnet'], 
                        help='Optimizee type')
    parser.add_argument('--features', type=int, default=300, help='Number of features for optimizee')
    parser.add_argument('--l1_weight', type=float, default=0.1, help='L1 weight for LASSO')
    parser.add_argument('--noise', type=float, default=0.1, help='Noise level')
    parser.add_argument('--data',type=int, default=100, help='Number of data points')
    parser.add_argument('--subset_size', type=int, default=1000, help='Subset size for each agent')
    parser.add_argument('--per_agent_batch_size', type=int, default=100, help='Batch size for each agent')
    
    # Save and device
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--save_path', type=str, default='./model.pth', help='Path to save model')
    parser.add_argument('--save_intermediate', action='store_true', 
                        help='Save model after each training stage')
    
    return parser

def parse_args():
    """
    Parse command line arguments
    """
    parser = get_parser()
    args = parser.parse_args()
    
    # If a stages configuration file is provided, load it
    if args.stages_config:
        with open(args.stages_config, 'r') as f:
            stages_config = json.load(f)
            
        # Update the stages parameter based on the config file
        args.stages = len(stages_config)
        args.stages_data = stages_config
    else:
        # Create a single-stage configuration using the command line arguments
        args.stages_data = [{
            'lr': args.lr,
            'truncation_length': args.truncation_length,
            'iterations': args.iterations,
            'epochs': args.epochs
        }]
    
    return args