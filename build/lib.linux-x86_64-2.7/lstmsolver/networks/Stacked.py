#A combination classifier by stacking two networks together
__author__="Thomas Breuel"

from Network import *


class Stacked(Network):
    '''Stack two networks on top of each other'''
    def __init__(self,nets):
       self.nets=nets 
       self.dstats=defaultdict(list)
    def ninputs(self):
        return self.nets[0].ninputs()
    def noutputs(self):
        return self.nets[-1].noutputs()
    def forward(self,xs):
        for i,net in enumerate(self.nets):
            xs = net.forward(xs)
        return xs
    def backward(self,deltas):
        self.ldeltas = [deltas]
        for i,net in reversed(list(enumerate(self.nets))):
            if deltas is not None:
                self.dstats[i].append((amin(deltas),mean(deltas),amax(deltas)))
            deltas = net.backward(deltas)
            self.ldeltas.append(deltas)
        self.ldeltas = self.ldeltas[::-1]
        return deltas
    def lastdeltas(self):
        return self.ldeltas[-1]
    def info(self):
        for net in self.nets:
            net.info()
    def states(self):
        return self.nets[0].states()
    def weights(self):
        for i,net in enumerate(self.nets):
            for w,dw,n in net.weights():
                yield w,dw,"Stacked%d/%s"%(i,n)
