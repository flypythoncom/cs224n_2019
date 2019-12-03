#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1h
import torch
import numpy as np
class Highway(torch.nn.Module):
    def __init__(self, D_in, H, D_out,prob):
        """
        Apply the output of the convolution later (x_conv) through a highway network
                @param D_in (int): Size of input layer
                @param H (int): Size of Hidden layer
                @param D_out (int): Size of output layer
                @param prob (float): Probability of dropout
        """
        super(Highway, self).__init__()


    def forward(self, x):
        """
        Apply the output of the convolution later (x_conv) through a highway network
                @param x (Tensor): Input x_cov gets applied to Highway network - shape of input tensor [batch_size,1,e_word]
                @returns x_pred (Tensor): Size of Hidden layer -- NOTE: check the shapes
        """
        pass


### END YOUR CODE 

