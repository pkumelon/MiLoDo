import torch 



class MiLoDo():
    def __init__(self, optimizee, max_iter, device, model,mlp_now = None,resnet_now = None):
        self.optimizee = optimizee
        self.device = device
        self.max_iter = max_iter
        self.model = model
        
        # Initialize tensors
        self.X = torch.zeros(1, optimizee.n, optimizee.d).to(device)
        
        if mlp_now is not None:
            for i in range(optimizee.n):
                self.X[0][i] = mlp_now.clone().to(self.device)
        if resnet_now is not None:
            for i in range(optimizee.n):
                self.X[0][i] = resnet_now.clone().to(self.device)
        
        self.Y = torch.zeros(1, optimizee.n, optimizee.d).to(device)
        self.GX = torch.zeros(1, optimizee.n, optimizee.d).to(device)
        self.Z = torch.zeros(1, optimizee.n, optimizee.d).to(device)
        self.lasso_weight = optimizee.l1_weight

    def run(self):
        loss = []
        
        for i in range(self.max_iter):
            self.GX[0] = self.optimizee.gradient(self.X[0])
            
            with  torch.no_grad():
                self.X, self.Y, self.Z = self.model(self.X, self.Y, self.Z, self.GX, self.lasso_weight)
            loss_now = self.optimizee.loss(self.X[0]).item()
            loss.append(loss_now)
            print("MiLoDo: iteration: %d, loss: %.10f" % (i, loss_now))
                
        return loss  # Return loss history