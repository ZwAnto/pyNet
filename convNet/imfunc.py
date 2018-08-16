import numpy as np
import time 
import pickle
import matplotlib.pyplot as plt

### Zeros padding function ###
def impad(x, pad):
    x_pad = np.zeros((x.shape[0],x.shape[1]+pad*2,x.shape[2]+pad*2,x.shape[3]))
    x_pad[:,pad:x.shape[1]+1,pad:x.shape[2]+1,:] = x
    
    return x_pad

### Image to column function ###
def im2col(x,size,pad,stride):
    
    # Padding image if required
    x = impad(x,pad)

    # Input shape
    N, H, W, C = x.shape

    # Computing indices for each dimensions H,W,C
    n = int((H - size)/stride + 1)
    n2= n**2
    base = [i*stride for i in list(range(n))]
    
    i = np.tile(np.repeat(range(size),C*size),n2) + np.repeat(base,size**2*C*n)
    j = np.tile(np.tile(np.repeat(range(size),C),size),n2) + np.tile(np.repeat(base,size**2*C),n)
    k = np.tile(range(C),size**2*n2)
    
    return x[:,i,j,k].reshape(N*n2,-1)

### Columns to image function ###
def col2im(x_col,shape,size,pad,stride):

    # Output Shape
    N, H, W, C = shape

    # Computing indices
    n = int((H - size)/stride + 1)
    n2= n**2
    base = [i*stride for i in list(range(n))]
  
    i = np.tile(np.repeat(range(size),C*size),n2) + np.repeat(base,size**2*C*n)
    j = np.tile(np.tile(np.repeat(range(size),C),size),n2) + np.tile(np.repeat(base,size**2*C),n)
    k = np.tile(range(C),size**2*n2)
    
    # Output array
    x_out = np.zeros(shape)
    x_out_c = np.zeros(shape)
    x_col_reshape = x_col.reshape(N,C*size**2*n2)

    # Column to array aggregation
    #np.add.at(x_out, (slice(None), i, j, k), x_col_reshape)

    # loop is faster than np.add.at
    for ii in range(len(i)):
        x_out[:,i[ii],j[ii],k[ii]] += x_col_reshape[:,ii]
        x_out_c[:,i[ii],j[ii],k[ii]] += 1
    
    # Should we avegrage the aggregate output ? Apparently not...
    #x_out /= x_out_c
    
    # Subseting output if there is any padding
    if pad == 0:
        return x_out

    return x_out[:, padding:-padding, padding:-padding, :]

### Softmax score function ####
def softmax(xfc,y,train=True):
    score = xfc
    score -= score.max(axis=1).reshape((-1,1))
    p = np.exp(score) / np.sum(np.exp(score),axis=1).reshape((-1,1))

    l = -np.log(p[np.arange(len(p)),y]+1e-20).sum() /len(y)

    acc = (sum(np.argmax(xfc.reshape(len(y),10),axis=1) == y)/len(y)*100)

    if (train):
        p[np.arange(len(p)), y] -= 1

    return p,l,acc

### Saving/Loading function ####
def save_object(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)
def load_object(filename):
    with open(filename, 'rb') as input:
        model = pickle.load(input)
    return(model)

### Plot training ###
def plot_training(model):
    fig, ax = plt.subplots(2,2)
    ax1 = ax[1,0]
    ax2 = ax1.twinx()
    ax3 = ax[0,0]
    ax4 = ax[0,1]

    ax[1,1].axis('off')

    x=[i/450 for i in range(len(model.batchAcc))]

    ln11=ax1.plot(x,model.batchAcc,c="grey")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")

    ln12=ax2.plot(x,model.batchLoss,c="deepskyblue")
    ax2.set_ylabel("Loss")

    obj = ln11+ln12
    ax2.legend(obj, ["Accuracy","Loss"], loc='upper left')

    ax1.set_title('Batch accuracy and loss')

    ln31 = ax3.plot(model.epochAcc,c="grey")
    ln32 = ax3.plot(model.testAcc,c="deepskyblue")
    ax3.set_xlabel('epoch')
    ax3.set_ylabel('accuracy')
    obj3 = ln31+ln32
    ax3.legend(obj3,['train', 'test'], loc='upper left')
    ax3.set_title('Train and test accuracy')


    ln41 = ax4.plot(model.epochLoss,c="grey")
    ln42 = ax4.plot(model.testLoss,c="deepskyblue")
    ax4.set_xlabel('epoch')
    ax4.set_ylabel('loss')
    obj4 = ln41+ln42
    ax4.legend(obj4,['train', 'test'], loc='upper left')
    ax4.set_title('Train and test loss')

    fig.set_figheight(6)
    fig.set_figwidth(10)
    fig.tight_layout()
    plt.show()

### Weird stuff ####
def logo():
    print(" ▄████▄   ▄▄▄       ██▀███     ▄▄▄█████▓▓█████ ▄▄▄       ███▄ ▄███▓")
    print("▒██▀ ▀█  ▒████▄    ▓██ ▒ ██▒   ▓  ██▒ ▓▒▓█   ▀▒████▄    ▓██▒▀█▀ ██▒")
    print("▒▓█    ▄ ▒██  ▀█▄  ▓██ ░▄█ ▒   ▒ ▓██░ ▒░▒███  ▒██  ▀█▄  ▓██    ▓██░")
    print("▒▓▓▄ ▄██▒░██▄▄▄▄██ ▒██▀▀█▄     ░ ▓██▓ ░ ▒▓█  ▄░██▄▄▄▄██ ▒██    ▒██ ")
    print("▒ ▓███▀ ░ ▓█   ▓██▒░██▓ ▒██▒     ▒██▒ ░ ░▒████▒▓█   ▓██▒▒██▒   ░██▒")
    print("░ ░▒ ▒  ░ ▒▒   ▓▒█░░ ▒▓ ░▒▓░     ▒ ░░   ░░ ▒░ ░▒▒   ▓▒█░░ ▒░   ░  ░")
    print("  ░  ▒     ▒   ▒▒ ░  ░▒ ░ ▒░       ░     ░ ░  ░ ▒   ▒▒ ░░  ░      ░")
    print("░          ░   ▒     ░░   ░      ░         ░    ░   ▒   ░      ░   ")
    print("░ ░            ░  ░   ░                    ░  ░     ░  ░       ░   ")
    print("░                                                                  ")
    return
