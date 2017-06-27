# SimpleNeuralNetwork

Simple Fully Connected Neural Network implemented in Matlab

Objective : To understand effect of hyper-parameters (Dropout, Learning Rate, Momentum, Weight Decay) on neural network convergence and generalization. 

Dataset: MNIST Subset<br />  
Training Set : 3000 Samples [300 x 10]<br />
Validation Set : 1000 Samples [100 x 10]<br />
Testing Set : 3000 [300 x 10]<br />
  
Model Architecture:<br />
  Inputs [784] -> FCN1[500] -> Sigmoid -> Dropout[0.5] -> FCN2[500] -> Sigmoid -> Dropout[0.5] -> Outputs[10]

Implementation:<br />
  1. Forward Prop, Back Prop Algorithm with SGD with modularity.

Architecture Variations:
  1. Number of layers in the architecture can be changed by adjusting the params in 'run.m' file
  2. Activation Units can be changed by specifying the model architecture in 'define_model.m' 
  3. Loss is Cross Entroy Error Function (Negative Log-Likelihood Function)
  4. Other params : Learning Rate, Momentum, Weight Decay, Dropout, Epochs
 
Results:
  
 ![Alt text](https://cloud.githubusercontent.com/assets/5204400/18943193/1a442784-85ec-11e6-8a7f-3907c4c578cb.jpg)
 ![Alt text](https://cloud.githubusercontent.com/assets/5204400/18943413/74678ea8-85ed-11e6-8ac7-dba50fe330a6.jpg)
 
 Using  all  the  hyper-parameters  mentioned  in  above  table  gave  best  validation  accuracy  (94%).   Testaccuracy with this model is 92.7%
 
Hyper - Parameter Effects:<br />
1. Error Rate w.r.t multiple Learning Rate values 

![Alt text] (https://cloud.githubusercontent.com/assets/5204400/18943453/ac472806-85ed-11e6-994e-828381608da8.jpg)

2. Error Rate w.r.t Hidden Units  
![Alt text] (https://cloud.githubusercontent.com/assets/5204400/18943462/bc4c2256-85ed-11e6-88b5-b32d26f85fee.jpg)
3. Error Rate w.r.t Dropout Values
![Alt text] (https://cloud.githubusercontent.com/assets/5204400/18943459/b6617972-85ed-11e6-8008-e93bf301b6ee.jpg)


 
