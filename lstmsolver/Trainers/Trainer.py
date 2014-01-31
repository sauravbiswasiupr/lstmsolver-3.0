#!/usr/bin/python 
#This is the main Trainer class for any kind of network that you need to train on
from numpy import *
from pylab import *


class Trainer():
    '''Trainer class that uses a network that is specified and enables the learning of the data'''
    def __init__(self,net,lr,train_set_x,train_set_y,val_set_x,val_set_y,width,height,epochs=100):
        '''initialize the Trainer params'''
        self.net = net 
        self.lr=lr 
        self.net.setLearningRate(self.lr) #momentum set automatically by this function call. We can change it later
        self.train_set_x=train_set_x
        self.train_set_y=train_set_y
        self.val_set_x=val_set_x
        self.val_set_y=val_set_y
        self.epochs=epochs  #100  by default 
        self.width=width
        self.height=height
        
    def train(self):
        tr_inps=self.train_set_x
        tr_targets=self.train_set_y
        val_inps=self.val_set_x
        val_targets=self.val_set_y
        tr_length=tr_inps.shape[0]
        tr_indices=[i for i in range(tr_length)]
        val_length=val_inps.shape[0] 
        val_indices=[i for i in range(val_length)]
        train_errors=[]
        valid_errors=[]
        best_valid_error=9999  # a large number
        maxtests=30
        tests=0
        for epoch in range(self.epochs):
             sum_error_tr=0
             sum_error_val=0
             print "Epoch : " , epoch 
             #shuffle indices
             shuffle(tr_indices)
             shuffle(val_indices)
             for i in tr_indices:
                  img=tr_inps[i]
                  target=tr_targets[i]
                  target=target.reshape((self.height,self.width))
                  img=img.reshape((self.height,self.width))
                  pred=array(self.net.forward(img))
                  deltas=target-pred
                  error=sum(deltas**2)/float(self.width*self.height)
                  sum_error_tr=sum_error_tr+error
                  a=self.net.backward(deltas)
                  self.net.update()
             #now for the validation set error 
             
             for i in val_indices:
                        img=val_inps[i] 
                        target=val_targets[i] 
                        target=target.reshape((self.height,self.width))
                        img=img.reshape((self.height,self.width))
                        pred=array(self.net.forward(img))
                        deltas=target-pred
                        error=sum(deltas**2)/float(self.width*self.height)
                        sum_error_val=sum_error_val+error
                        
             sum_error_tr=sum_error_tr/float(len(tr_indices))
             sum_error_val=sum_error_val/float(len(val_indices))
             print "Epoch ", epoch , "Training error : " , sum_error_tr , "|| Validation Error : " ,sum_error_val
             train_errors.append(sum_error_tr)
             valid_errors.append(sum_error_val)
             print "Tests now :" , tests 
             if tests > maxtests:
                 break 
             
             if sum_error_val<best_valid_error :
                            tests=0
                            best_valid_error=sum_error_val
                            print "Best validation error : ", best_valid_error
                            sum_error_val=0 
                            
             else:
                tests=tests+1                
             
        return train_errors,valid_errors
            
    def predict(self,x):
        '''Just the standard forward pass'''
        return array(self.net.forward(x))
