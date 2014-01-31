#!/usr/bin/python
'''Implementation of a Sigmoid Layer'''
__author__="Saurav Biswas"

#fix imports
from Network import *

class SigmoidLayer(Network):
    '''A Sigmoid Layer that can be used as a hidden layer or an output layer'''
    def __init__(self,Nh,No,initial_range=initial_range,rand=rand):
        self.Nh=Nh
        self.No=No 
        self.W=randu(No,Nh+1)*initial_range
        self.DW=zeros((No,Nh+1))
    def ninputs(self):
        return self.Nh 
    def noutputs(self):
        return self.No 
    def forward(self,ys):
        #the forward pass function 
        n=len(ys)
        inputs,zs = [None]*n,[None]*n
        for i in range(n):
            inputs[i] = concatenate([ones(1),ys[i]])
            zs[i] = sigmoid(dot(self.W,inputs[i]))
        self.state = (inputs,zs)
        return zs
    def backward(self,deltas):
        inputs,zs = self.state
        n = len(zs)
        assert len(deltas)==len(inputs)
        dzspre,dys = [None]*n,[None]*n
        for i in reversed(range(len(zs))):
            dzspre[i] = deltas[i] * zs[i]*(1-zs[i])
            dys[i] = dot(dzspre[i],self.W)[1:]
        self.dzspre = dzspre
        self.DW = sumouter(dzspre,inputs)
        return dys
    def info(self):
        vars = sorted("W".split())
        for v in vars:
            a = array(getattr(self,v))
            print v,a.shape,amin(a),amax(a)
    def weights(self):
        yield self.W,self.DW,"SigmoidLayer"
