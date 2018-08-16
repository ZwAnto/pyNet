
### CLASS : relu layer ###

class relu:
 
    # Initialization of object
    def __init__(self):
        return
    
    # Forward pass
    def forward(self,X):
        X_ = X
        X_[X<0]*=0
        
        self.x = X
        return X_
    
    # Backward pass
    def backward(self,dot):
        dot[self.x<= 0] *= 0
        #dot = np.multiply(dot,d_relu)
        return dot
