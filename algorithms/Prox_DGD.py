import torch

### DGD
class DGD:
    def __init__(self, optimizee, learning_rate, iterations, device, topo_A,mlp_now = None,resnet_now = None):
        self.optimizee = optimizee
        self.learning_rate = learning_rate
        self.device = device
        self.max_iterations = iterations
        
        # Initialize dimensions from optimizee
        self.n = optimizee.n
        self.d = optimizee.d
        
        # Initialize algorithm variables
        self.X = torch.zeros(self.n, self.d).to(device)
        
        if mlp_now is not None:
            for i in range(self.n):
              self.X[i] = mlp_now.clone().to(self.device)
        
        if resnet_now is not None:
            for i in range(self.n):
              self.X[i] = resnet_now.clone().to(self.device)
        
        self.G = torch.zeros(self.n, self.d).to(device)
        
        # Topology matrix
        self.A = topo_A/3
        self.A = self.A.to(device)
        
        # Learning rate diagonal matrix
        self.matrix = torch.ones(self.d, device=self.device) * self.learning_rate
        
    def run(self):
        loss_history = []
        
        for i in range(self.max_iterations):
            # Compute gradients
            self.G = self.optimizee.gradient(self.X)
            
            # Gradient descent step
            Y = self.X - self.learning_rate*0.5 * self.G
            
            # Proximal operator
            Z = torch.zeros_like(self.X, device=self.device)
            for j in range(self.n):
                Z[j,:] = self.optimizee.prox(self.matrix, Y[j,:])
            
            # Consensus step
            self.X = self.A @ Z
            
            # Record loss
            current_loss = self.optimizee.loss(self.X).item()
            loss_history.append(current_loss)
            
            # Print progress
            print(f"DGD: iteration: {i}, loss: {current_loss:.10f}")
        
        return loss_history