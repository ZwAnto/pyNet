import numpy as np
import convNet.imfunc as imf

### CLASS : convolution layer ###

class conv:
    
    # Initialization of object
    def __init__ (self,pad,stride,fsize,cout):
        # Setting pad, stride, filter size and # of output channel
        self.pad = pad
        self.stride = stride
        self.fsize = fsize
        self.cout = cout
        
    # Forward pass
    def forward(self,x):

        # Input shape
        self.xshape = x.shape
        N,W,H,C = x.shape
        
        # Number of time we apply filter along H or W
        n = int((W+2*self.pad - self.fsize)/self.stride + 1)

        # If weight array is not set
        if not hasattr(self,'w'):
            
            ### Weight initialization ###
            
            # Different test: random normal, uniform, ...
            #self.w = np.random.randn(self.cout,self.fsize,self.fsize,C)* np.sqrt(2.0/(self.fsize*self.fsize*C))
            #self.w = np.random.randn(self.cout,self.fsize,self.fsize,C)* np.sqrt(2.0/(self.cout*self.fsize*self.fsize*C))
            #self.w = np.random.randn(self.cout,self.fsize,self.fsize,C) * 0.01
            #self.w = np.random.uniform(size=(self.cout,self.fsize,self.fsize,C),low=-0.05,high=0.05)
            #self.w = np.random.normal(0,0.05,(self.cout,self.fsize,self.fsize,C))
            
            # Glorot / Xavier initialization
            fan_in = C*self.fsize*self.fsize
            fan_out = self.cout*self.fsize*self.fsize
            limit = np.sqrt(6 / (fan_in + fan_out )) 
            self.w = np.random.uniform(size=(self.cout,self.fsize*self.fsize*C),low=-limit,high=limit)
            
            ### Bias initialization ###
            self.b = np.zeros(self.cout)
            
            # Moving mean of dw and db initializes to zero
            self.cache_w = np.zeros((self.cout,self.fsize*self.fsize*C))
            self.cache_b = np.zeros((self.cout))
            
        # Image to column    
        self.x_col = imf.im2col(x,self.fsize,self.pad,self.stride)

        # Dot product
        dot = self.x_col @ self.w.T + self.b

        # Output reshaping
        dot_reshape = dot.reshape(N,n,n,self.cout)

        return dot_reshape

    # Backward pass
    def backward(self,dot,l_rate):
        
        # w and output shape
        w_shape = self.w.shape
        x_out_shape = self.xshape
        
        # Reshaping gradient
        dot = dot.reshape(self.x_col.shape[0],-1)
        
        # Weight gradient
        dw = (self.x_col.T.dot(dot)).T
    
        # Bias gradient
        db = dot.sum(axis=0)
        
        # X gradient
        dot = dot.dot(self.w)
        dot = imf.col2im(dot,x_out_shape,self.fsize,self.pad,self.stride)

        # Weight update : RMSprop
        self.cache_w = 0.9* self.cache_w + (1 - 0.9) * dw**2
        update = - l_rate * dw / (np.sqrt(self.cache_w) + 1e-20)
        #update = - l_rate*dw
        self.w += update# the actual update

        # Bias update : RMSprop
        self.cache_b = 0.9 * self.cache_b + (1 - 0.9) * db**2
        update = - l_rate * db / (np.sqrt(self.cache_b) + 1e-20)
        #update = - l_rate*db
        self.b += update
        
        return dot
 
    
