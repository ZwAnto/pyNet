import numpy as np
import time
from sklearn.utils import shuffle

import convNet.imfunc as imf
import convNet.batchnorm as bn
import convNet.conv as cv
import convNet.fully as fc
import convNet.maxpool as mp
import convNet.relu as rl
import convNet.dropout as dp


class conv_net:
    def __init__ (self):
        self.init = ""
        self.forward_func = ""
        self.predict_func = ""
        self.backward_func = ""
        self.n_cv = -1
        
        self.batchLoss = []
        self.batchAcc = []
        self.epochLoss = []
        self.epochAcc = []
        self.testLoss = []
        self.testAcc = []

        return
    
    def add_conv(self,pad,stride,fsize,c):
        self.n_cv += 1
        self.init += "self.cv_" + str(self.n_cv) + " = cv.conv(" + str(pad) + "," + str(stride) + "," + str(fsize) + "," + str(c) + ")\n"
        
        self.forward_func += "x = self.cv_" + str(self.n_cv) + ".forward(x)\n"
        self.predict_func += "x = self.cv_" + str(self.n_cv) + ".forward(x)\n"

        self.backward_func = "dot = self.cv_" + str(self.n_cv) + ".backward(dot=dot,l_rate=l_rate)\n" + self.backward_func
        
        return
    
    def add_relu(self):
        self.init += "self.rl_" + str(self.n_cv) + " = rl.relu()\n"
        
        self.forward_func += "x = self.rl_" + str(self.n_cv) + ".forward(x)\n" 
        self.predict_func += "x = self.rl_" + str(self.n_cv) + ".forward(x)\n" 
        self.backward_func = "dot = self.rl_" + str(self.n_cv) + ".backward(dot)\n" + self.backward_func
        return
        
    def add_maxpool(self,size):
        self.init += "self.mp_" + str(self.n_cv) + " = mp.maxpool(" + str(size) + ")\n"
        
        self.forward_func += "x = self.mp_" + str(self.n_cv) + ".forward(x)\n"
        self.predict_func += "x = self.mp_" + str(self.n_cv) + ".forward(x)\n"
        self.backward_func = "dot = self.mp_" + str(self.n_cv) + ".backward(dot)\n" + self.backward_func
        return
    
    def add_batchnorm(self):
        self.init += "self.bn_" + str(self.n_cv) + " = bn.batchnorm()\n"
        self.forward_func += "x = self.bn_" + str(self.n_cv) + ".forward(x)\n"
        self.predict_func += "x = self.bn_" + str(self.n_cv) + ".forward(x)\n"
        self.backward_func = "dot = self.bn_" + str(self.n_cv) + ".backward(dot=dot,l_rate=l_rate)\n" + self.backward_func
        return
    
    def add_fully(self,size):
        self.init += "self.fc_" + str(self.n_cv) + " = fc.fully(" + str(size) + ")\n"
        
        self.forward_func += "x = self.fc_" + str(self.n_cv) + ".forward(x)\n"
        self.predict_func += "x = self.fc_" + str(self.n_cv) + ".forward(x)\n"
        self.backward_func = "dot = self.fc_" + str(self.n_cv) + ".backward(dot=dot,l_rate=l_rate)\n" + self.backward_func
        return
    
    def add_dropout(self,p):
        self.init += "self.dp_" + str(self.n_cv) + " = dp.dropout(" + str(p) + ")\n"
        
        self.forward_func += "x = self.dp_" + str(self.n_cv) + ".forward(x)\n"
        self.backward_func = "dot = self.dp_" + str(self.n_cv) + ".backward(dot=dot)\n" + self.backward_func
        
        return
    
    def compile(self):
        exec(self.init)
    
    def convnet_backward(self,dot,l_rate):
        exec(self.backward_func)
        
    def convnet_forward(self,x):
        env={}
        exec(self.forward_func,locals(),env)
        
        return env['x']
    
    def predict(self,xin,y,scale=True):
        x = np.copy(xin)
        if scale:
            #x[:, self.Xstd>0] /= self.Xstd[self.Xstd>0]
            x[:, self.Xstd>0] = np.true_divide(x[:,self.Xstd>0],self.Xstd[self.Xstd>0])
        
        env={}
        exec(self.predict_func,locals(),env)
        p,l,acc = imf.softmax(env['x'],y,train=False)
    
        return (p,l,acc)
    
    def train(self,train,test,l_rate,b_size,n_epoch):
                
        imf.logo()
        
        X_, Y_ = train
        X_test_, Y_test_ = test
        
        X = np.copy(X_)
        Y = np.copy(Y_)
        X_test = np.copy(X_test_)
        Y_test = np.copy(Y_test_)
        
        self.Xstd = np.std(X,axis = 0)
        X[:, self.Xstd>0] /= self.Xstd[self.Xstd>0]
        X_test[:, self.Xstd>0] /= self.Xstd[self.Xstd>0]
        
        n_iter = X.shape[0]/b_size
        
        start_time = time.time()

        # EPOCH LOOP START
        for epoch in range(n_epoch):

            ### SHUFFLE TRAINING SET ###############################
            X, Y = shuffle(X, Y)

            # BATCH LOOP START
            for i in range(int(n_iter)): 

                ### BATCH ##############################################

                X_ = X[(i*b_size):((i+1)*b_size)]
                Y_ = Y[(i*b_size):((i+1)*b_size)]

                ### FORWARD PASS #######################################

                xfc = self.convnet_forward(X_)

                ### SOFTMAX SCORE FUNCTION #############################

                dot,l,acc = imf.softmax(xfc,Y_)

                self.batchLoss.append(l)
                self.batchAcc.append(acc)

                ### BACKWARD PASS ######################################

                self.convnet_backward(dot,l_rate)

                ### OUTPUT TEXT ########################################

                if (i+1)%10== 0:
                    size=20
                    pad = int((i+1)/n_iter*size)

                    last=">"
                    if (i+1) == n_iter:
                        last = "="

                    text = "Epoch " + str(epoch) + " - " + str(i+1).zfill(len(str(int(n_iter)))) + "/" + str(int(n_iter)) + \
                          " " + "█" * pad + "░" * int(size-pad) +" acc: " \
                          + str(round(np.mean(self.batchAcc[-10:]),3)) + " loss: " + str(round(np.mean(self.batchLoss[-10:]),8))
                    #clear_output()
                    print(text) 

            # BATCH LOOP END

            ### TEST ACCURACY AND LOSS ############################

            p, l, acc = self.predict(X_test,Y_test,scale=False)
            
            self.testLoss.append(l)
            self.testAcc.append(acc)

            self.epochLoss.append(np.mean(self.batchLoss[-b_size:]))
            self.epochAcc.append(np.mean(self.batchAcc[-b_size:]))
            print("Epoch loss: " +  str(np.mean(self.batchLoss[-b_size:])))
            print("Epoch acc: " +  str(np.mean(self.batchAcc[-b_size:])))
            print("Test loss: " +  str(l))
            print("Test acc: " +  str(acc))

        # EPOCH LOOP END

        end_time = time.time()
