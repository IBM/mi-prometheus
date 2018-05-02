#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ntm_model.py: Factory building models"""
__author__ = "Tomasz Kornuta"

import torch 


class NTM(torch.nn.Module):
    '''  Class representing the Neural Turing Machine module. '''
    def __init__(self, params):
        '''
        Constructor. Initializes parameters on the basis of dictionary of parameters passed as argument .
        
        :param params: Dictionary of parameters.
        '''
        # Call constructor of base class.
        super(NTM, self).__init__() 

        # Set input and  dimensions.
        self.in_dim = params["control_bits"] + params["data_bits"]
        self.out_dim = params["data_bits"]
        self.ctrl_hidden_dim = params['ctrl_hidden_dim']

        self.linear1 = torch.nn.Linear(self.in_dim, self.ctrl_hidden_dim)
        self.linear2 = torch.nn.Linear(self.ctrl_hidden_dim, self.out_dim)

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary (differentiable) operations on Tensors.
        """
        h_relu = self.linear1(x).clamp(min=0)
        y_pred = self.linear2(h_relu)
        return y_pred


if __name__ == "__main__":
    # "Loaded parameters".
    params = {'control_bits': 5, 'data_bits': 995, 'batch_size': 64,  'ctrl_hidden_dim': 100}
        
    in_dim = params["control_bits"] + params["data_bits"]
    out_dim = params["data_bits"]
        
    # Create random Tensors to hold inputs and outputs
    x = torch.randn(params['batch_size'], in_dim)
    y = torch.randn(params['batch_size'], out_dim)

    # Construct our model by instantiating the class defined above
    model = NTM(params)

    # Construct our loss function and an Optimizer. The call to model.parameters()
    # in the SGD constructor will contain the learnable parameters of the two
    # nn.Linear modules which are members of the model.
    criterion = torch.nn.MSELoss(size_average=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
    for t in range(500):
        # Forward pass: Compute predicted y by passing x to the model
        y_pred = model(x)

        # Compute and print loss
        loss = criterion(y_pred, y)
        print(t, loss.item())

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    


