import torch



class Prox_ATC():
    def __init__(self, optimizee, lr, max_iter,topo_A,mlp_now = None,resnet_now = None):
        self.optimizee = optimizee
        self.device = optimizee.device
        self.lr = lr
        self.max_iter = max_iter
        self.n = optimizee.n
        self.d = optimizee.d
        self.phi = torch.zeros((self.n, self.d), device=optimizee.device) ## Intermediate variable phi
        
        if mlp_now is not None:
            for i in range(self.n):
              self.phi[i] = mlp_now.clone().to(self.device)
        
        if resnet_now is not None:
            for i in range(self.n):
              self.phi[i] = resnet_now.clone().to(self.device)
        
        self.X = self.phi.clone() ## X to be optimized, initialized as phi
        self.W1 = torch.zeros((self.n, self.d), device=optimizee.device) ## i
        self.W2 = torch.zeros((self.n, self.d), device=optimizee.device) # i-1
        self.W3 = torch.zeros((self.n, self.d), device=optimizee.device) # i-2
        self.Z = torch.zeros_like(self.X,device = self.device) ## Intermediate variable
        self.A = topo_A/3
        self.A = self.A.to(self.device)
        self.gradient1 = torch.zeros_like(self.X,device = self.device) ## Gradient i 
        self.gradient2 = torch.zeros_like(self.X,device = self.device) # i-1
        self.gradient3 = torch.zeros_like(self.X,device = self.device) # i-2
        ## Create a vector with all values equal to lr
        self.matrix = torch.ones( self.d,device=self.device) * self.lr

    def run(self):
        loss = []
        for i in range(self.max_iter):
            self.gradient1 = self.optimizee.gradient(self.W1)
            phi_mid=self.phi.clone()
            self.phi = self.W1 - self.lr * self.gradient1
            self.Z = 2* self.X - self.A @ (self.X - self.phi+ phi_mid)
            self.X = self.A @ self.Z    
            for j in range(self.n):
                self.W1[j,:] = self.optimizee.prox(self.matrix ,self.X[j,:])
            loss.append(self.optimizee.loss(self.W1).item())     
            print ("Prox_ATC_I: iteration:",i," loss:",self.optimizee.loss(self.X).item()) 
     
        return loss

     
