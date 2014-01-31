#!/usr/bin/python 
from numpy import *
from pylab import *

def confusion_matrix(predicted,target.nclasses):
   '''Accept lists containing predicted outputs and another array containing targets. Return the confusion matrix'''
   length1=len(predicted) 
   length2=len(target) 
   assert length1==length2
   tp=0
   tn=0
   fp=0
   fn=0
   for i in range(length1):
       if predicted[i] == target[i]:
           #true positive 
           tp=tp+1 
       
