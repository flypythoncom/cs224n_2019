#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1i
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
class CNN(nn.Module):
    def __init__(self, in_ch, out_ch,k=5):
        """ 
        Apply the output of the convolution later (x_conv) through a highway network
                @param D_in (int): Size of input layer 
                @param H (int): Size of Hidden layer
                @param D_out (int): Size of output layer
                @param prob (float): Probability of dropout
        """
        super(CNN, self).__init__()
     


    def forward(self, x):
        pass
### END YOUR CODE

