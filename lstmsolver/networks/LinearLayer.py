'''An implementation of a Linearlayer. There is no transfer function '''
__author__="Saurav Biswas"
#fix imports 

from Network import *

class LinearLayer(Network): 
    '''A Linear Layer that has no activation function'''
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
            zs[i] = dot(self.W,inputs[i])
        self.state = (inputs,zs)
        return zs
    def backward(self,deltas):
        inputs,zs = self.state
        n = len(zs)
        assert len(deltas)==len(inputs)
        dzspre,dys = [None]*n,[None]*n
        for i in reversed(range(len(zs))):
            dzspre[i] = deltas[i] 
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
        yield self.W,self.DW,"LinearLayer"
