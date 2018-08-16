
<img src='https://cdn0.iconfinder.com/data/icons/isometric-city-basic-transport/48/car-front-01-128.png' />

# LINCOLN DataScience Challenge #2
## CIFAR-10 - Image Recongnition
---

# Content of the directory
---
- **convNet** folder: it's a python package containing all the functions and class relative to the custom implementation of the CNN.
- **MNIST** and **CIFAR-10** folders: contains the training datasets.
- **ConvNet-MNIST.ipynb**: notebook that train a CNN using the custom implementation on the MNIST dataset.
- **ResNet-CIFAR-10.ipynb**: notebook that train a Residual Network using Keras on the CIFAR-10 dataset.

# Module requirements
---
For the custom implementation and the residual network
- numpy
- sklearn
- pickle
- matplotlib
- time
- scipy

For the residual network
- keras 
- tensorflow

# Importing datasets
---
### MNIST
Place the 'train.csv' corresponding to the MNIST dataset in the 'MNIST' folder.
### CIFAR-10
Place the 'train' folder and the 'trainLabels.csv' containing the training image and labels in the 'CIFAR-10' folder.

# How to use the custom implementation of CNN
---
To train a neural network using the custom implementation, you first need to load the **conv_net** class present in the **convNet.conv_net** module.


```python
from convNet.conv_net import conv_net
```


```python
import convNet.conv_net
```

Then you must initiate your model


```python
model = conv_net()
```

Then set-up your CNN architecture using the different implemented layers


```python
# Convolutionnal layer
model.add_conv(padding, # number of pixel to add around the image (zero-padding),
                        # not sure it actually works... keep it to 0
               stride, 
               filter_size,
               output_size)

# Relu layer
model.add_relu()

# Batchnorm layer
model.add_batchnorm()

# Maxpool layer
model.add_maxpool(filter_size)

# Dropout layer
model.add_dropout(p)

# Fully connected layer
model.add_fully(n)
```

Finnaly compile your model and train it!


```python
# X must be shapped like (N,WIDTH,HEIGHT,CHANNEL)
# Y must be a numeric list of class (shape: (N))

model.compile()
model.train(train=(X,Y),
            test=(X_test,Y_test),
            b_size=100, # batch size
            l_rate=0.001, # learning rate
            n_epoch=20 # Epoch
            )
```

You can save and reload your model using this functions (base on pickle module)


```python
import convNet.imfunc as imf

# to make a backup
imf.save_object(model, 'model_MNIST.pkl')

# to reload a model
model = imf.load_object('model_MNIST.pkl')
```

A more complete exemple is given in **ConvNet-MNIST.ipynb**.
