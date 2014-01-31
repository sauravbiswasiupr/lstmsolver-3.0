#!/usr/bin/python 
'''Script to have some useful utility functions to use later'''
from numpy import *
from pylab import * 
 
initial_range=0.1 

class RangeError(Exception):
   def __init__(self,s=None):
     Exception.__init__(self,s)


def randu(*shape):
    # ATTENTION: whether you use randu or randn can make a difference.
    """Generate uniformly random values in the range (-1,1).
    This can usually be used as a drop-in replacement for `randn`
    resulting in a different distribution."""
    return 2*rand(*shape)-1

def sigmoid(x):
    """Compute the sigmoid function.
    We don't bother with clipping the input value because IEEE floating
    point behaves reasonably with this function even for infinities."""
    return 1.0/(1.0+exp(-x))

def rownorm(a):
    """Compute a vector consisting of the Euclidean norm of the
    rows of the 2D array."""
    return sum(array(a)**2,axis=1)**.5

def check_nan(*args,**kw):
    "Check whether there are any NaNs in the argument arrays."
    for arg in args:
        if isnan(arg).any():
            raise FloatingPointError()

def sumouter(us,vs,lo=-1.0,hi=1.0,out=None):
    """Sum the outer products of the `us` and `vs`.
    Values are clipped into the range `[lo,hi]`.
    This is mainly used for computing weight updates
    in logistic regression layers."""
    result = zeros((len(us[0]),len(vs[0])))
    for u,v in zip(us,vs):
        result += outer(clip(u,lo,hi),v)
    return result

def sumprod(us,vs,lo=-1.0,hi=1.0,out=None):
    """Sum the element-wise products of the `us` and `vs`.
    Values are clipped into the range `[lo,hi]`.
    This is mainly used for computing weight updates
    in logistic regression layers."""
    assert len(us[0])==len(vs[0])
    result = zeros(len(us[0]))
    for u,v in zip(us,vs):
        result += clip(u,lo,hi)*v
    return result

def ffunc(x):
    "Nonlinearity used for gates."
    return 1.0/(1.0+exp(-x))
def fprime(x,y=None):
    "Derivative of nonlinearity used for gates."
    if y is None: y = sigmoid(x)
    return y*(1.0-y)
def gfunc(x):
    "Nonlinearity used for input to state."
    return tanh(x)
def gprime(x,y=None):
    "Derivative of nonlinearity used for input to state."
    if y is None: y = tanh(x)
    return 1-y**2
# ATTENTION: try linear for hfunc
def hfunc(x):
    "Nonlinearity used for output."
    return tanh(x)
def hprime(x,y=None):
    "Derivative of nonlinearity used for output."
    if y is None: y = tanh(x)
    return 1-y**2
