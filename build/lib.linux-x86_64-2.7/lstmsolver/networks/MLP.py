##A Multilayer Perceptron 
__author__="Thomas Breuel"

#fix imports 
from numpy import *
from pylab import *
from Network import *


class MLP(Network):
    """A multilayer perceptron (direct implementation)."""
    def __init__(self,Ni,Nh,No,initial_range=initial_range,rand=randu):
        self.Ni = Ni
        self.Nh = Nh
        self.No = No
        self.W1 = rand(Nh,Ni+1)*initial_range
        self.W2 = rand(No,Nh+1)*initial_range
    def ninputs(self):
        return self.Ni
    def noutputs(self):
        return self.No
    def forward(self,xs):
        n = len(xs)
        inputs,ys,zs = [None]*n,[None]*n,[None]*n
        for i in range(n):
            inputs[i] = concatenate([ones(1),xs[i]])
            ys[i] = sigmoid(dot(self.W1,inputs[i]))
            ys[i] = concatenate([ones(1),ys[i]])
            zs[i] = sigmoid(dot(self.W2,ys[i]))
        self.state = (inputs,ys,zs)
        return zs
    def backward(self,deltas):
        xs,ys,zs = self.state
        n = len(xs)
        dxs,dyspre,dzspre,dys = [None]*n,[None]*n,[None]*n,[None]*n
        for i in reversed(range(len(zs))):
            dzspre[i] = deltas[i] * zs[i] * (1-zs[i])
            dys[i] = dot(dzspre[i],self.W2)[1:]
            dyspre[i] = dys[i] * (ys[i] * (1-ys[i]))[1:]
            dxs[i] = dot(dyspre[i],self.W1)[1:]
        self.DW2 = sumouter(dzspre,ys)
        self.DW1 = sumouter(dyspre,xs)
        return dxs
    def weights(self):
        yield self.W1,self.DW1,"MLP1"
        yield self.W2,self.DW2,"MLP2"
