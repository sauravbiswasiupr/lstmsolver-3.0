#!/usr/bin/python
'''This file contains all the possible trainers for Sequence classification mode '''

__author__="Saurav Biswas"

#fix imports 
from numpy import *
from pylab import *

def make_target(targetval,nclasses,img_height):
    target=zeros((nclasses,))
    target[targetval]=1
    target_over_time=zeros((img_height,nclasses))
    for i in range(img_height):
        target_over_time[i,:]=target
    return target_over_time

class Trainer_2D_LSTM():
    '''Trainer for Seq classification mode using a 2D LSTM'''
    def __init__(self,net,lr,train_set_x,train_set_y,val_set_x,val_set_y,width,height,nclasses,epochs=100,autosave=False):
        '''initialize the Trainer params '''
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
        self.nclasses=nclasses
        self.autosave=autosave
        
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
                  #target=target.reshape((self.height,self.width))
                  #target=hstack((target,target))
                  target_over_time = make_target(target,self.nclasses,self.height)
                  img=img.reshape((self.height,self.width))
                  pred=array(self.net.forward(img))
                  deltas=target_over_time-pred
                  #error=sum(deltas**2)/float(self.height*self.nclasses)
                  #error=sum(deltas[-1]**2)
                  prediction = argmax(pred[-1]) 
                  if prediction != target:
                       sum_error_tr=sum_error_tr+1
                  a=self.net.backward(deltas)
                  self.net.update()
             #now for the validation set error 
             
             for i in val_indices:
                        img=val_inps[i] 
                        target=val_targets[i] 
                        #target=target.reshape((self.height,self.width))
                        #target=hstack((target,target))
                        target_over_time=make_target(target,self.nclasses,self.height)
                        img=img.reshape((self.height,self.width))
                        pred=array(self.net.forward(img))
                        deltas=target_over_time-pred
                        #error=sum(deltas**2)/float(self.height*self.nclasses)
                        #error=sum(deltas[-1]**2)
                        prediction=argmax(pred[-1])
                        if prediction != target:
                                sum_error_val=sum_error_val+1
                        
             sum_error_tr=sum_error_tr/float(len(tr_indices))
             sum_error_val=sum_error_val/float(len(val_indices))
             #print "Epoch ", epoch , "Training error : " , sum_error_tr , "|| Validation Error : " ,sum_error_val
             s="Epoch :" +str(epoch)+" Training Error :" + str(sum_error_tr)+" || Validation Error : " + str(sum_error_val)
             print s
             if self.autosave==True:
                 f=open('errors.save','a')
                 f.write(s+'\n') 
             train_errors.append(sum_error_tr)
             valid_errors.append(sum_error_val)
             print "Tests now :" , tests 
             if tests > maxtests:
                 break 
             
             if sum_error_val<best_valid_error  :
                            tests=0
                            best_valid_error=sum_error_val
                            print "Best validation error : ", best_valid_error
                            sum_error_val=0 
                            
             else:
                tests=tests+1                
        print "Best validation error was : " , best_valid_error 
        cmd='Best validation error:' + str(best_valid_error)
        f.write(cmd+'\n')
        return train_errors,valid_errors
 
    def predict_image(self,x):
        pred=array(self.net.forward(x))
        return argmax(pred[-1])
