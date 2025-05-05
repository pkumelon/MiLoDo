import torch



class Prox_ED():
    def __init__(self, optimizee, lr, max_iter,topo_A,mlp_now = None,resnet_now = None):
        self.optimizee = optimizee
        self.device = optimizee.device
        self.lr = lr
        self.max_iter = max_iter
        self.n = optimizee.n
        self.d = optimizee.d
        self.phi = torch.zeros((self.n, self.d), device=optimizee.device) 
        
        if mlp_now is not None:
            for i in range(self.n):
              self.phi[i] = mlp_now.clone().to(self.device)
        
        if resnet_now is not None:
            for i in range(self.n):
              self.phi[i] = resnet_now.clone().to(self.device)
        
        self.X = self.phi.clone() 
        self.W = torch.zeros_like(self.X,device = self.device) 
        self.Z = torch.zeros_like(self.X,device = self.device) 
        
        
    
        self.A_bar = (torch.eye(self.n) + topo_A/3) / 2
        
        self.A_bar = self.A_bar.to(self.device)
        self.gradient = torch.zeros_like(self.X,device = self.device)

        self.matrix = torch.ones( self.d,device=self.device) * self.lr


    def run(self):
        loss = []
        for i in range(self.max_iter):
            self.gradient = self.optimizee.gradient(self.W)
            for j in range(self.n):
                phi_i = self.phi[j, :].clone() 
                self.phi[j,:]=self.W[j,:]-self.lr*self.gradient[j,:]
                self.Z[j,:] = self.X[j,:] + self.phi[j,:] - phi_i
            
            self.X = self.A_bar @ self.Z    
            
            for j in range(self.n):
                self.W[j,:] = self.optimizee.prox(self.matrix ,self.X[j,:])
            loss.append(self.optimizee.loss(self.W).item())    
            print ("Prox_ED: iteration:",i," loss:",self.optimizee.loss(self.W).item())     
        return loss    