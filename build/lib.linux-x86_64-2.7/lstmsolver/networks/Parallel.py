#A network Wrapper that runs several networks in parallel
#Useful when we need something like a several networks running in several directions in several orientations

__author__="Thomas Breuel"

from Network import *

class Parallel(Network):
    """Run multiple networks in parallel on the same input."""
    def __init__(self,*nets):
        self.nets = nets
    def forward(self,xs):
        outputs = [net.forward(xs) for net in self.nets]
        outputs = zip(*outputs)
        outputs = [concatenate(l) for l in outputs]
        return outputs
    def backward(self,deltas):
        deltas = array(deltas)
        start = 0
        for i,net in enumerate(self.nets):
            k = net.noutputs()
            net.backward(deltas[:,start:start+k])
            start += k
        return None
    def info(self):
        for net in self.nets:
            net.info()
    def states(self):
        states = [net.states() for net in self.nets]
        outputs = zip(*outputs)
        outputs = [concatenate(l) for l in outputs]
        return outputs
    def weights(self):
        for i,net in enumerate(self.nets):
            for w,dw,n in net.weights():
                yield w,dw,"Parallel%d/%s"%(i,n)
