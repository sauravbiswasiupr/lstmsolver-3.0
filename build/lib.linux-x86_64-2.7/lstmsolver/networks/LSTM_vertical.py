#!/usr/bin/python 
'''This following module creates an LSTM network that works on inputs in the y direction.
That is from +y to -y. This is especially useful when you need to combine with the standard LSTM 
that works in the horizontal direction'''
from LSTM import *
from utilities import *

class LSTM_vertical(LSTM):
   def forward(self,xs):
        """Perform forward propagation of activations"""
        xs=xs.T
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
