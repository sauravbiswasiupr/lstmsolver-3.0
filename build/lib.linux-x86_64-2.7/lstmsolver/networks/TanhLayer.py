'''An implementation of a tanhlayer '''
__author__="Saurav Biswas"
#fix imports 

from Network import *

class TanhLayer(Network): 
    '''A tanh layer that can be added as a hidden layer with tanh activations'''
    def __init__(self,Nh,No,initial_range=initial_range,rand=rand):
        self.Nh=Nh
        self.No=No
        self.W2=randu(No,Nh+1)*initial_range
        self.DW2=zeros((No,Nh+1))
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
            zs[i] = tanh(dot(self.W2,inputs[i]))
        self.state = (inputs,zs)
        return zs
    def backward(self,deltas):
        inputs,zs = self.state
        n = len(zs)
        assert len(deltas)==len(inputs)
        dzspre,dys = [None]*n,[None]*n
        for i in reversed(range(len(zs))):
            dzspre[i] = deltas[i] * (1-zs[i]**2)
            dys[i] = dot(dzspre[i],self.W2)[1:]
        self.dzspre = dzspre
        self.DW2 = sumouter(dzspre,inputs)
        return dys
    def info(self):
        vars = sorted("W2".split())
        for v in vars:
            a = array(getattr(self,v))
            print v,a.shape,amin(a),amax(a)
    def weights(self):
        yield self.W2,self.DW2,"TanhLayer"
