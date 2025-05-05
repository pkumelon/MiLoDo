import torch 


class PG_EXTRA():
    def __init__(self, optimizee, lr, max_iter,topo_A,mlp_now = None,resnet_now = None):    
        self.optimizee = optimizee
        self.device = optimizee.device
        self.lr = lr
        self.max_iter = max_iter
        self.n = optimizee.n
        self.d = optimizee.d
        
        self.X1 = torch.zeros((self.n, self.d), device=optimizee.device) ## X t-1
        
        if mlp_now is not None:
            for i in range(self.n):
              self.X1[i] = mlp_now.clone().to(self.device)
        
        if resnet_now is not None:
            for i in range(self.n):
              self.X1[i] = resnet_now.clone().to(self.device)
        
        self.X2 = self.X1.clone() ## X t-1/2
        self.X3 = self.X1.clone() ## X t
        self.X4 = self.X1.clone() ## X t+1/2
        self.X5 = self.X1.clone() ## X t+1
        
        self.A = topo_A/3
        self.A_bar = (torch.eye(self.n) + self.A) / 2
        self.A = self.A.to(self.device) 
        self.A_bar = self.A_bar.to(self.device)
        
        self.gradient1 = torch.zeros_like(self.X1,device = self.device) ## Gradient X t-1
        self.gradient3 = self.gradient1.clone() ## Gradient X t
        
        self.matrix = torch.ones( self.d,device=self.device) * self.lr ## Create a vector with all values equal to lr


    def run(self):
        loss = []
        for i in range(self.max_iter):
            self.gradient1 = self.optimizee.gradient(self.X1)
            self.gradient3 = self.optimizee.gradient(self.X3)
            
            
            if i>0:
                self.X4 = self.A @ self.X3 + self.X2 - self.A_bar @ self.X1 -self.lr *(self.gradient3 - self.gradient1)
                self.X5 = self.optimizee.prox(self.matrix, self.X4)
                
                ## Update X
                self.X1 = self.X3.clone()
                self.X2 = self.X4.clone()
                self.X3 = self.X5.clone()
   
            else:
                self.X4 = self.A @ self.X3  -self.lr *(self.gradient3)
                self.X5 = self.optimizee.prox(self.matrix, self.X4)
                
                ## Update X
                self.X1 = self.X3.clone()
                self.X2 = self.X4.clone()
                self.X3 = self.X5.clone()

                    
            
            loss_now = self.optimizee.loss(self.X5).item()
            loss.append(loss_now)
            
            ## Print loss for this generation
            print('PG-EXTRA: iter:',i,'loss:',loss_now)

        return loss