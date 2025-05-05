import torch 

def generate_metropolis_weights(topo_A):
    A = topo_A
    n = A.shape[0]
    W = torch.zeros_like(A)
    for i in range(n):
        for j in range(n):
            if A[i, j] > 0:
                W[i, j] = 1 / (1 + max(torch.sum(A[i]), torch.sum(A[j])))

    for i in range(n):
        W[i, i] = 1 - torch.sum(W[i]) + W[i, i]
        
    # print(W)    
    return W

class ODAPG:
    def __init__(self, optimizee, lr, max_iter, topo_A, K=1, tau=0.2,mlp_now = None,resnet_now = None):
        self.optimizee = optimizee
        self.device = optimizee.device
        self.lr_input = lr
        self.lr = lr
        self.tau = tau
        self.max_iter = max_iter
        self.n = optimizee.n
        self.d = optimizee.d
        self.K = K
        self.L = 0.01
        
        # Initialize variables
        self.X = torch.zeros((self.n, self.d), device=self.device)
        
        if mlp_now is not None:
            for i in range(self.n):
              self.X[i] = mlp_now.clone().to(self.device)
        
        if resnet_now is not None:
            for i in range(self.n):
              self.X[i] = resnet_now.clone().to(self.device)
        
        self.Y = self.X.clone()
        self.Z = self.X.clone()
        self.S = self.optimizee.gradient(self.X).clone()

        # Generate mixing matrix
        self.A_bar = generate_metropolis_weights(topo_A)
        self.A_bar = self.A_bar.to(self.device)

    def fastmix(self, x, W, K):
        for _ in range(K):
            x = torch.matmul(W, x)
        return x

    def run(self):
        loss = []

        for t in range(1, self.max_iter + 1):
            
            # Compute x_{t+1}
            X_next = self.tau * self.Z + (1 - self.tau) * self.Y
            
            # Compute local gradients and update s_{t+1}
            grad_X_next = self.optimizee.gradient(X_next)
            S_next = self.fastmix(self.S + grad_X_next - self.optimizee.gradient(self.X), self.A_bar, self.K)
            
            # Update z_{t+1}
            Z_next = self.fastmix(self.optimizee.prox(self.lr, self.Z - self.lr*0.85 * S_next), self.A_bar, self.K)
            
            # Update y_{t+1}
            Y_next = self.fastmix(self.tau * Z_next + (1 - self.tau) * self.Y, self.A_bar, self.K)
            
            # Update the variables
            self.X = X_next
            self.Y = Y_next
            self.Z = Z_next
            self.S = S_next
            
            current_loss = self.optimizee.loss(self.X).item()
            loss.append(current_loss)
            
    
            # Print the current iteration's loss and consensus error
            print(f"ODAPG: iteration: {t}, loss: {current_loss:.20f}")

        return loss