This library is a fast implementation of standard neural network architectures namely Multilayer Perceptrons and its variants. It also implements networks networks such as RNNs and LSTMs which act on time sequence inputs. 

Installation Instructions
-------------------------

Assuming you have extracted the tarball to an appropriate directory. Just issue the following command in the terminal 

```$ sudo python setup.py install ```

This will install in the default directory /usr/local/python2.7/dist-packages/


If you want to do a local install use:

```$python setup.py install --prefix=/usr/local```



General Instructions:

* To create an LSTM:
```
>> from lstmsolver.networks import LSTM
>>lstm1=LSTM(ni,nh)
```
Note that the parameters ni,nh can be image height and width for classification mode

* For time reversed LSTM
```
>>from lstmsolver.networks import Reversed,Stacked,Parallel
>>lstm2=Reversed(LSTM(ni,nh))
```
 
* For two networks in parallel
```
>>bidi=Parallel(lstm1,lstm2,lstm3...)  
```

I also included a vertical LSTM that kind of works on the input in the vertical direction that is -y to +y. Just include it using

```
>>from lstmsolver.networks import LSTM_vertical
>>lstm_y_forward=LSTM_vertical(Ni,Nh) 
>>lstm_y_backward=Reversed(LSTM_vertical(Ni,Nh))
```
and do Parallel as usual 


* To combine all the four lstms in x and y directions do 
```
>>blstm_2d=Parallel(lstm1,lstm2,lstm_y_forward,lstm_y_backward) 
```

now we need a softmax layer, so :
`>>softmax=Softmax(Nh,No)  #where No=number of classes`

You would need a Logreg layer for binary classification (logit regression) 
that would be :
`>>logreg=Logreg(Nh,1) `
now stack up the layers 

`>>network=Stacked([blstm_2d,softmax]) `


Now all you need is a trainer, for SC mode the trainer is available as Trainer_2D_LSTM from Trainers module 
`>>from lstmsolver.Trainers import Trainer_2D_LSTM`
now trainer class needs several params, some are included by default,
The trainers train() fn returns the training and validation errors over all the epochs 
`>>trainer=Trainer_2D_LSTM(network,train_x,train_y,valid_x,valid_y,img_width,img_height,nclasses,epochs=numepochs,autosave=True) `
  
So for a dataset with images of size say (28,28) and 10 classes (MNIST)and say for 100 epochs you would instantiate the trainer as 
```
>>trainer=Trainer_2D_LSTM(network,train_x,train_y,valid_x,valid_y,28,28,10,epochs=100,autosave=True) 
>>tr,val=trainer.train() 
```
starts the training and the output is saved to 'errors.save' in the same directory that you start the experiment from 

##Added a tanhlayer that can be stacked with any network as a hidden or output layer. It has tanh nonlinearities (transfer function) 

##Added a SigmoidLayer and a LinearLayer that work as name suggests
