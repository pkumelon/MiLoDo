<p align="center">
  <img src="assets/logo1.png" alt="MiLoDo Logo" width="60%" style="max-width: 300px;"/>
</p>

# MiLoDo: A Mathematics-Inspired Learning-to-Optimize Framework for Decentralized Optimization

MiLoDo is an innovative framework for decentralized optimization that leverages meta-learning techniques to enhance the performance and convergence speed of distributed optimization algorithms. The framework provides advanced solutions for various optimization problems, including L1 regularized problems and more complex neural network optimizations (MLP and ResNet).

## Features

- **Adaptive Meta-learning Optimizer**: Trained through meta-learning to outperform traditional decentralized optimization algorithms
- **Multiple Optimization Problems**: Support for LASSO regression, L1 logistic regression, MLP neural networks, and ResNet
- **Algorithm Comparisons**: Implementation of 7 decentralized optimization algorithms (MiLoDo, Prox-DGD, Prox-ED, Prox-ATC, PG-EXTRA, DAPG, ODAPG)
- **Multi-stage Training**: Flexible multi-stage training process through configuration files
- **Results Visualization**: Intuitive comparison of different algorithms' performance

## Dependencies

```bash
pip install -r requirements.txt
```

## File Structure

```
MiLoDo/
├── algorithms/          # Distributed optimization algorithm implementations
├── data/                # Dataset storage directory
├── fig/                 # Result charts output directory
├── MiLoDo/              # Core MiLoDo framework
├── model/               # Directory for saved models
├── optimizees/          # Optimization problem definitions
├── scripts/             # Training scripts and configurations
├── utils/               # Utility functions
├── test.py              # Testing and comparison script
└── train.py             # Training script
```

## Training Tutorial

The MiLoDo framework uses a multi-stage training strategy to achieve optimal performance, with flexible training parameters adjustable through configuration files.

### 1. Configure Training Parameters

Training parameters are defined in the `scripts/stages_config.json` file, containing multiple training stages:

```json
[
    {
        "lr": 0.0005,              // Learning rate
        "truncation_length": 5,    // Inner steps
        "iterations": 10,          // Outer iteration count
        "epochs": 20               // Training epochs per stage
    },
    // More stage configurations...
]
```

Parameter meanings for each stage:
- `lr`: Meta-optimizer learning rate
- `truncation_length`: Truncated backpropagation length (inner steps)
- `iterations`: Optimization iterations per batch (outer steps)
- `epochs`: Number of training epochs for the current stage

### 2. Run Training Script

Use the provided shell script to easily start training:

```bash
chmod +x scripts/run_training.sh   # Add execution permission
./scripts/run_training.sh          # Run with default parameters
```

Or directly use Python:

```bash
python train.py --stages_config "./scripts/stages_config.json" \
    --batch_size 2 \
    --agents 10 \
    --optimizee lasso \
    --features 300 \
    --l1_weight 0.1 \
    --noise 0.1 \
    --use_bias \
    --lr_init 0.03 \
    --optimizer adam \
    --weight_decay 1e-4 \
    --seed 42 \
    --save_path "./model/multi_stage_model.pth" \
    --save_intermediate
```

### 3. Main Training Parameters

- `--optimizee`: Optimization problem type (`lasso`, `logistic`, `mlp`, `resnet`)
- `--agents`: Number of agents (nodes) in the network
- `--batch_size`: Batch size for meta-learning
- `--features`: Feature dimension (LASSO/Logistic regression)
- `--l1_weight`: L1 regularization weight
- `--noise`: Data noise level
- `--save_path`: Path to save the final model
- `--save_intermediate`: Save intermediate stage models

### 4. Training Output

The training process will save the model to the path specified by `--save_path`. If `--save_intermediate` is enabled, an intermediate model will be saved for each stage. Training information is saved as a JSON file (same name as the model file with the suffix `_training_info.json`).

## Testing Tutorial

The testing script (`test.py`) is used to compare the performance of MiLoDo with other decentralized optimization algorithms.

### 1. Configure Testing Parameters

Testing parameters are configured directly in the `test.py` file:

```python
# Basic settings
seed = 42
num_agents = 10

# Algorithm parameters
iterations = 100
learning_rate = 0.01

# Optimization problem type
optimizee_type = 'lasso'  # Options: 'lasso', 'logistic', 'mlp', 'resnet'

# MiLoDo model settings
load_pretrained = False
model_path = "./model/multi_stage_model.pth"
```

Main configuration items:
- `optimizee_type`: Select optimization problem type
- `iterations`: Number of iterations for comparison testing
- `learning_rate`: Learning rate for baseline algorithms
- `load_pretrained`: Whether to load a pretrained MiLoDo model
- `model_path`: Path to the pretrained model

### 2. Run Test

Run the test script directly:

```bash
python test.py
```

### 3. Test Results

The test will compare the performance of 7 decentralized optimization algorithms and generate result charts:
- MiLoDo 
- Prox-DGD
- Prox-ED
- Prox-ATC
- PG-EXTRA
- DAPG
- ODAPG

Result charts will be saved in the `./fig/` directory, with the format `{optimization_problem_type}-{number_of_nodes}agents.png`.

## Citation


```
@misc{he2024mathematicsinspiredlearningtooptimizeframeworkdecentralized,
      title={A Mathematics-Inspired Learning-to-Optimize Framework for Decentralized Optimization}, 
      author={Yutong He and Qiulin Shang and Xinmeng Huang and Jialin Liu and Kun Yuan},
      year={2024},
      eprint={2410.01700},
      archivePrefix={arXiv},
      primaryClass={math.OC},
      url={https://arxiv.org/abs/2410.01700}, 
}
```
