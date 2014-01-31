#!/usr/bin/python
#fix imports 
from numpy import *
from pylab import *

from Network import *
from utilities import *

def randu(*shape):
    # ATTENTION: whether you use randu or randn can make a difference.
    """Generate uniformly random values in the range (-1,1).
    This can usually be used as a drop-in replacement for `randn`
    resulting in a different distribution."""
    return 2*rand(*shape)-1


class LSTM(Network):
    """A standard LSTM network. This is a direct implementation of all the forward
    and backward propagation formulas, mainly for speed. (There is another, more
    abstract implementation as well, but that's significantly slower in Python
    due to function call overhead.)"""
    def __init__(self,ni,ns,initial=1,maxlen=5000):
        na = 1+ni+ns
        self.dims = ni,ns,na
        self.init_weights(initial)
        self.allocate(maxlen)


    def ninputs(self):
        return self.dims[0]


    def noutputs(self):
        return self.dims[1]


    def states(self):
        """Return the internal state array for the last forward
        propagation. This is mostly used for visualizations."""
        return array(self.state[:self.last_n])


    def init_weights(self,initial):
        "Initialize the weight matrices and derivatives"
        ni,ns,na = self.dims
        # gate weights
        for w in "WGI WGF WGO WCI".split():
            setattr(self,w,randu(ns,na)*initial)
            setattr(self,"D"+w,zeros((ns,na)))
        # peep weights
        for w in "WIP WFP WOP".split():
            setattr(self,w,randu(ns)*initial)
            setattr(self,"D"+w,zeros(ns))


    def weights(self):
        "Yields all the weight and derivative matrices"
        weights = "WGI WGF WGO WCI WIP WFP WOP"
        for w in weights.split():
            yield(getattr(self,w),getattr(self,"D"+w),w)


    def info(self):
        "Print info about the internal state"
        vars = "WGI WGF WGO WIP WFP WOP cix ci gix gi gox go gfx gf"
        vars += " source state output gierr gferr goerr cierr stateerr"
        vars = vars.split()
        vars = sorted(vars)
        for v in vars:
            a = array(getattr(self,v))
            print v,a.shape,amin(a),amax(a)

    def allocate(self,n):
        """Allocate space for the internal state variables.
        `n` is the maximum sequence length that can be processed."""
        ni,ns,na = self.dims
        vars = "cix ci gix gi gox go gfx gf"
        vars += " state output gierr gferr goerr cierr stateerr outerr"
        for v in vars.split():
            setattr(self,v,nan*ones((n,ns)))
        self.source = nan*ones((n,na))
        self.sourceerr = nan*ones((n,na))


    def reset(self,n):
        """Reset the contents of the internal state variables to `nan`"""
        vars = "cix ci gix gi gox go gfx gf"
        vars += " state output gierr gferr goerr cierr stateerr outerr"
        vars += " source sourceerr"
        for v in vars.split():
            getattr(self,v)[:,:] = nan


    def forward(self,xs):
        """Perform forward propagation of activations."""
        ni,ns,na = self.dims
        assert len(xs[0])==ni
        n = len(xs)
        if n>len(self.gi): 
	    raise ocrolib.RecognitionError("input too large for LSTM model")
        self.last_n = n
        self.reset(n)
        for t in range(n):
            prev = zeros(ns) if t==0 else self.output[t-1]
            self.source[t,0] = 1
            self.source[t,1:1+ni] = xs[t]
            self.source[t,1+ni:] = prev
            dot(self.WGI,self.source[t],out=self.gix[t])
            dot(self.WGF,self.source[t],out=self.gfx[t])
            dot(self.WGO,self.source[t],out=self.gox[t])
            dot(self.WCI,self.source[t],out=self.cix[t])
            if t>0:
                # ATTENTION: peep weights are diagonal matrices
                self.gix[t] += self.WIP*self.state[t-1]
                self.gfx[t] += self.WFP*self.state[t-1]
            self.gi[t] = ffunc(self.gix[t])
            self.gf[t] = ffunc(self.gfx[t])
            self.ci[t] = gfunc(self.cix[t])
            self.state[t] = self.ci[t]*self.gi[t]
            if t>0:
                self.state[t] += self.gf[t]*self.state[t-1]
                self.gox[t] += self.WOP*self.state[t]
            self.go[t] = ffunc(self.gox[t])
            self.output[t] = hfunc(self.state[t]) * self.go[t]
        assert not isnan(self.output[:n]).any()
        return self.output[:n]


    def backward(self,deltas):
        """Perform backward propagation of deltas."""
        n = len(deltas)
        if n>len(self.gi): 
	    raise ocrolib.RecognitionError("input too large")
        assert n==self.last_n
        ni,ns,na = self.dims
        for t in reversed(range(n)):
            self.outerr[t] = deltas[t]
            if t<n-1:
                self.outerr[t] += self.sourceerr[t+1][-ns:]
            self.goerr[t] = fprime(None,self.go[t]) * hfunc(self.state[t]) * self.outerr[t]
            self.stateerr[t] = hprime(self.state[t]) * self.go[t] * self.outerr[t]
            self.stateerr[t] += self.goerr[t]*self.WOP
            if t<n-1:
                self.stateerr[t] += self.gferr[t+1]*self.WFP
                self.stateerr[t] += self.gierr[t+1]*self.WIP
                self.stateerr[t] += self.stateerr[t+1]*self.gf[t+1]
            if t>0:
                self.gferr[t] = fprime(None,self.gf[t])*self.stateerr[t]*self.state[t-1]
            self.gierr[t] = fprime(None,self.gi[t])*self.stateerr[t]*self.ci[t] # gfunc(self.cix[t])
            self.cierr[t] = gprime(None,self.ci[t])*self.stateerr[t]*self.gi[t]
            dot(self.gierr[t],self.WGI,out=self.sourceerr[t])
            if t>0:
                self.sourceerr[t] += dot(self.gferr[t],self.WGF)
            self.sourceerr[t] += dot(self.goerr[t],self.WGO)
            self.sourceerr[t] += dot(self.cierr[t],self.WCI)
        self.DWIP = nutils.sumprod(self.gierr[1:n],self.state[:n-1],out=self.DWIP)
        self.DWFP = nutils.sumprod(self.gferr[1:n],self.state[:n-1],out=self.DWFP)
        self.DWOP = nutils.sumprod(self.goerr[:n],self.state[:n],out=self.DWOP)
        self.DWGI = nutils.sumouter(self.gierr[:n],self.source[:n],out=self.DWGI)
        self.DWGF = nutils.sumouter(self.gferr[1:n],self.source[1:n],out=self.DWGF)
        self.DWGO = nutils.sumouter(self.goerr[:n],self.source[:n],out=self.DWGO)
        self.DWCI = nutils.sumouter(self.cierr[:n],self.source[:n],out=self.DWCI)
        return [s[1:1+ni] for s in self.sourceerr[:n]]

