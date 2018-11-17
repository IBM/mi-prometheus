LeNet-5 Model
-------------

Now, let us focus on the LeNet-5 model.

is a convolutional neural network (CNN). CNNs are the foundation of modern state-of-the art deep learning-based computer vision.

.. figure:: ../img/lenet5_architecture.png
    :figwidth: 100 %
    :align: center
    
    Examples of MNIST images with corresponding labels (targets).





Source: Y.LeCun, L.Bottou, Y.Bengio, and P.Haffner. `Gradient-based learning applied to document recognition
<http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf>`_, śProc. IEEE 86(11): 2278–2324, 1998. 


It only has 7 layers, among which there are 3 convolutional layers (C1, C3 and C5), 2 sub-sampling (pooling) layers (S2 and S4), 
and 1 fully connected layer (F6), that are followed by the output layer. 
Convolutional layers use 5 by 5 convolutions with stride 1. Sub-sampling layers are 2 by 2 average pooling layers. Tanh sigmoid activations are used throughout the network.



We will derive that model directly from the base Model class.
Analogically, in __init__, we define the data_definitions – here, we specify the image size the model expects and the shape of the predictions (logits) that it will return. 
Next, we create all the model’s trainable layers. The second important method is forward, which computes the logits on the basis of the inputs, by simply passing outputs from one layer to the other.
