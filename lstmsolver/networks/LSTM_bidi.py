#A typical implementation of a 1D bidirectional LSTM
#need top just import LSTM_vertical which imports other reqd stuff into the namespace automatically

__author__="Saurav Biswas"

from LSTM_vertical import *
from numpy import *
from pylab import *
from Stacked import Stacked


class LSTM_BIDI_1D(Network):
   '''A one dimensional LSTM that works on time sequenced inputs'''
   def __init__(self,net1,net2):
      self.net1=net1 
      self.net2=net2
      self.net=Stacked(net1,net2)
   
       
       
            





   
