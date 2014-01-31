#A network that runs on time reversed input. Primarily 
#used for RNNs or LSTMs or something that works on time sequences

__author__="Thomas Breuel"

from Network import *

class Reversed(Network):
    """Run a network on the time-reversed input."""
    def __init__(self,net):
        self.net = net
    def ninputs(self):
        return self.net.ninputs()
    def noutputs(self):
        return self.net.noutputs()
    def forward(self,xs):
        return self.net.forward(xs[::-1])[::-1]
    def backward(self,deltas):
        result = self.net.backward(deltas[::-1])
        return result[::-1] if result is not None else None
    def info(self):
        self.net.info()
    def states(self):
        return self.net.states()[::-1]
    def weights(self):
        for w,dw,n in self.net.weights():
            yield w,dw,"Reversed/%s"%n

