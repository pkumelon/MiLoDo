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


class DAPG:
    def __init__(self, optimizee, lr, max_iter, topo_A, K=1, alpha=0.25,mlp_now = None,resnet_now = None):
        self.optimizee = optimizee
        self.device = optimizee.device
        self.lr = lr
        self.max_iter = max_iter
        self.n = optimizee.n
        self.d = optimizee.d
        self.K = K
        self.alpha = alpha
        
        # Initialize variables
        self.X = torch.zeros((self.n, self.d), device=self.device)
        if mlp_now is not None:
            for i in range(self.n):
              self.X[i] = mlp_now.clone().to(self.device)
        
        if resnet_now is not None:
            for i in range(self.n):
              self.X[i] = resnet_now.clone().to(self.device)
        
        self.Y = self.X.clone()
        self.S = self.optimizee.gradient(self.X).clone()

        # Generate mixing matrix
        self.A_bar = generate_metropolis_weights(topo_A)
        self.A_bar = self.A_bar.to(self.device)

        # Create a vector filled with lr values
        self.matrix = torch.ones(self.d, device=self.device) * self.lr

    def fastmix(self, x, W, K):
        for _ in range(K):
            x = torch.matmul(W, x)
        return x

    def run(self):
        loss = []
        consensus_error = []
        for i in range(self.max_iter):
            # Update x
            X_next = self.fastmix(self.optimizee.prox(self.lr, self.Y - self.lr*0.5 * self.S), self.A_bar, self.K)
            # Update y
            Y_next = self.fastmix(X_next + (1 - self.alpha) / (1 + self.alpha) * (X_next - self.X), self.A_bar, self.K)
            # Update s
            S_next = self.fastmix(self.S + self.optimizee.gradient(Y_next) - self.optimizee.gradient(self.Y), self.A_bar, self.K)
            
            # Update the variables
            self.X = X_next
            self.Y = Y_next
            self.S = S_next
            
            current_loss = self.optimizee.loss(self.X).item()
            loss.append(current_loss)
            
            # Print the current iteration's loss and consensus error
            print(f"DAPG: iteration: {i}, loss: {current_loss:.20f}")

        return loss
    