import numpy as np

### CLASS : fully connected layer ###

class fully:
    
    # Initialization of object
    def __init__(self,size):
        self.size=size
        return

    # Forward pass
    def forward(self,x):
        
        # Input shape
        N,W,H,C = x.shape
        
        # Saving input in object
        self.x = x
        
         # If weight array is not set
        if not hasattr(self,'w'):
            
             ### Weight initialization ###
                
            # Different test: random normal, uniform, ...
            #self.w = np.random.randn(W*H*C,nclass)* np.sqrt(2.0/(W*H*C))
            #self.w = np.random.randn(W*H*C,nclass) * np.sqrt(2.0/(W*H*C*nclass))
            #self.w = np.random.randn(W*H*C,nclass) * 0.01
            #self.w = np.random.uniform(size=(W*H*C,nclass),low=-0.05,high=0.05)
            #self.w = np.random.normal(0,0.05,(W*H*C,nclass))
            
            # Glorot / Xavier initialization
            fan_in = C*W*H
            fan_out = self.size
            limit = np.sqrt(6 / (fan_in + fan_out))
            self.w = np.random.uniform(size=(W*H*C,self.size),low=-limit,high=limit)
            
            ### Bias initialization ###
            self.b = np.zeros(self.size)
            
            # Moving mean of dw and db initializes to zero
            self.cache_w = np.zeros((W*H*C,self.size))
            self.cache_b = np.zeros((self.size))
    
        return x.reshape((x.shape[0],-1))@self.w + self.b

    # Backward pass
    def backward(self,dot,l_rate):
        
        # Weight gradient
        dw = self.x.reshape(self.x.shape[0],-1).T.dot(dot)
        dw /= self.x.shape[0]
        
        # Bias gradient
        db = dot.sum(axis=0)
        dot = dot.dot(self.w.T).reshape(self.x.shape) 
        
        # Weight update : RMSprop
        self.cache_w = 0.99 * self.cache_w + (1 - 0.99) * dw**2
        update = - l_rate * dw / (np.sqrt(self.cache_w) + 1e-20)
        #update = - l_rate*dw
        self.w += update# the actual update

        # Bias update : RMSprop
        self.cache_b = 0.99 * self.cache_b + (1 - 0.99) * db**2
        update = - l_rate * db / (np.sqrt(self.cache_b) + 1e-20)
        #update = - l_rate*db
        self.b += update
        
        return dot

