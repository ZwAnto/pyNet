import numpy as np

### CLASS : max pool layer ###

class dropout:
 
    # Initialization of object
    def __init__(self,p):
        self.p = p
        return
    
    # Forward pass
    def forward(self,x):
        self.drop = (np.random.rand(*x.shape) < self.p) / self.p
        out = x * self.drop
        return out
    
    # Backward pass
    def backward(self,dot):
        dot *= self.drop
        return dot