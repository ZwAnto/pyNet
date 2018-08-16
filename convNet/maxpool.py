import numpy as np
import convNet.imfunc as imf

### CLASS : max pool layer ###

class maxpool:
 
    # Initialization of object
    def __init__(self,size):
        self.size = size
        return
    
    # Forward pass
    def forward(self,x):

        self.input_shape = x.shape
        
        # Input shape
        N, H, W, C = x.shape

        # Setting stride equal to size
        # Maxpool is like a convolution with size equal to stride
        stride=self.size

        # Number of time we apply filter along W or H
        n = int((H - self.size)/stride + 1)

        # Number of time we apply the filter along the whole image
        n2= n**2

        # Image to column
        x_col = imf.im2col(x,self.size,0,stride)
        x_col2 = x_col.reshape(N,n2,self.size**2,C)

        # Argmax... Because it's a maxpooling layer
        idx = x_col2.argmax(axis=2).flatten()

        # Computing gradient for backward pass
        d_max = (idx[:,None] == np.arange(idx.max()+1)).astype(float).T

        d_max = d_max.reshape(x_col2.shape[2],x_col2.shape[0],x_col2.shape[1],x_col2.shape[3]).transpose(1,2,0,3)

        d_max = imf.col2im(d_max,x.shape,self.size,0,self.size)

        self.d_max = d_max
        
        # Output reshaping
        out = x.reshape(N, int(H/self.size), self.size, int(W/self.size), self.size,C).max(axis=(2, 4))

        return out
    
    # Backward pass
    def backward(self,dot):

        dot_ = np.zeros(self.input_shape)

        for i in range(self.input_shape[0]):
            for j in range(self.input_shape[3]):
                for k in range(dot.shape[1]):
                    k_ = k*2
                    dot_[i,k_,:,j] = np.repeat(dot[i,k,:,j],2)
                    dot_[i,k_+1,:,j] = np.repeat(dot[i,k,:,j],2)

        dot_ = np.multiply(dot_,self.d_max)

        return dot_
    
    