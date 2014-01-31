#!/usr/bin/python 
#fix imports
from numpy import *
from pylab import *
from utilities  import *
import nutils
from collections import defaultdict


class Network:
    """General interface for networks. This mainly adds convenience
    functions for `predict` and `train`."""
    def predict(self,xs):
        return self.forward(xs)
    def train(self,xs,ys,debug=0):
        xs = array(xs)
        ys = array(ys)
        pred = array(self.forward(xs))
        deltas = ys - pred
        self.backward(deltas)
        self.update()
        return pred
    def ctrain(self,xs,cs,debug=0,lo=1e-5,accelerated=1):
        """Training for classification.  This handles
        the special case of just two classes. It also
        can use regular least square error training or
        accelerated training using 1/pred as the error signal."""
        assert len(cs.shape)==1
        assert (cs==array(cs,'i')).all()
        xs = array(xs)
        pred = array(self.forward(xs))
        deltas = zeros(pred.shape)
        assert len(deltas)==len(cs)
        # NB: these deltas are such that they can be used
        # directly to update the gradient; some other libraries
        # use the negative value.
        if accelerated:
            # ATTENTION: These deltas use an "accelerated" error signal.
            if deltas.shape[1]==1:
                # Binary class case uses just one output variable.
                for i,c in enumerate(cs):
                    if c==0:
                        deltas[i,0] = -1.0/max(lo,1.0-pred[i,0])
                    else:
                        deltas[i,0] = 1.0/max(lo,pred[i,0])
            else:
                # For the multi-class case, we use all output variables.
                deltas[:,:] = -pred[:,:]
                for i,c in enumerate(cs):
                    deltas[i,c] = 1.0/max(lo,pred[i,c])
        else:
            # These are the deltas from least-square error
            # updates. They are slower than `accelerated`,
            # but may give more accurate probability estimates.
            if deltas.shape[1]==1:
                # Binary class case uses just one output variable.
                for i,c in enumerate(cs):
                    if c==0:
                        deltas[i,0] = -pred[i,0]
                    else:
                        deltas[i,0] = 1.0-pred[i,0]
            else:
                # For the multi-class case, we use all output variables.
                deltas[:,:] = -pred[:,:]
                for i,c in enumerate(cs):
                    deltas[i,c] = 1.0-pred[i,c]
        self.backward(deltas)
        self.update()
        return pred
    def setLearningRate(self,r,momentum=0.9):
        """Set the learning rate and momentum for weight updates."""
        self.learning_rate = r
        self.momentum = momentum
    def weights(self):
        """Return an iterator that iterates over (W,DW,name) triples
        representing the weight matrix, the computed deltas, and the names
        of all the components of this network. This needs to be implemented
        in subclasses. The objects returned by the iterator must not be copies,
        since they are updated in place by the `update` method."""
        pass
    def allweights(self):
        """Return all weights as a single vector. This is mainly a convenience
        function for plotting."""
        aw = list(self.weights())
        weights,derivs,names = zip(*aw)
        weights = [w.ravel() for w in weights]
        derivs = [d.ravel() for d in derivs]
        return concatenate(weights),concatenate(derivs)
    def update(self):
        """Update the weights using the deltas computed in the last forward/backward pass.
        Subclasses need not implement this, they should implement the `weights` method."""
        if not hasattr(self,"verbose"):
            self.verbose = 0
        if not hasattr(self,"deltas") or self.deltas is None:
            self.deltas = [zeros(dw.shape) for w,dw,n in self.weights()]
        for ds,(w,dw,n) in zip(self.deltas,self.weights()):
            ds.ravel()[:] = self.momentum * ds.ravel()[:] + self.learning_rate * dw.ravel()[:]
            w.ravel()[:] += ds.ravel()[:]
            if self.verbose:
                print n,(amin(w),amax(w)),(amin(dw),amax(dw))
