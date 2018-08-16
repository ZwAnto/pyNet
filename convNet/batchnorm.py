import numpy as np

class batchnorm:
    
    # Initialization of object
    def __init__(self):
        self.eps = 1e-5
        
    # Forward pass
    def forward(self,x):
        if not hasattr(self,'gamma'):
            self.gamma = x.std(axis=0,keepdims=True)
            self.beta = x.mean(axis=0,keepdims=True)
            self.Eg = np.zeros(self.gamma.shape) 
            self.Eb = np.zeros(self.beta.shape) 
            self.mg = np.zeros(self.gamma.shape) 
            self.mb = np.zeros(self.beta.shape) 
            
        mu = x.mean(axis=0,keepdims=True)
        
        xmu = x-mu
        
        sq = xmu**2
        
        var = sq.sum(axis=0,keepdims=True) / x.shape[0]
        
        sqrtvar = np.sqrt(var + self.eps)
        
        ivar = 1./sqrtvar
        
        xhat = xmu * ivar
       
        gammax = self.gamma * xhat
        
        out = gammax + self.beta

        self.cache = (xhat,xmu,ivar,sqrtvar,var)

        return out
        
    # Backward pass
    def backward(self,dot,l_rate,beta1=0.9,beta2=0.999):
    
        xhat,xmu,ivar,sqrtvar,var = self.cache

        dbeta = dot.sum(axis=0)
        dgammax = dot

        dgamma = (dgammax*xhat).sum(axis=0)
        dxhat = dgammax*self.gamma

        divar = (dxhat*xmu).sum(axis=0,keepdims=True)
        dxmu1 = dxhat * ivar

        dsqrtvar = -1. /(sqrtvar**2) * divar

        dvar = 0.5 * 1. /np.sqrt(var+self.eps) * dsqrtvar

        dsq = 1. /xhat.shape[0] * np.ones(dot.shape) * dvar

        dxmu2 = 2 * xmu * dsq

        dx1 = (dxmu1 + dxmu2)
        dmu = -1 * np.sum(dxmu1+dxmu2, axis=0)

        dx2 = 1. /xhat.shape[0] * np.ones(dot.shape) * dmu

        dx = dx1 + dx2


        update = - l_rate*dgamma
        self.gamma += update# the actual update

    
        update = - l_rate*dbeta
        self.beta += update

        return dx