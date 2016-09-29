# SimpleNeuralNetwork

Simple Fully Connected Neural Network implemented in Matlab

Dataset: MNIST Subset<br />  
Training Set : 3000 Samples [300 x 10]<br />
Validation Set : 1000 Samples [100 x 10]<br />
Testing Set : 3000 [300 x 10]<br />
  
Model Architecture:

  Inputs [784] -> FCN1[500] -> Sigmoid -> Dropout[0.5] -> FCN2[500] -> Sigmoid -> Dropout[0.5] -> Outputs[10]

Architecture Variations:
  1. Number of layers in the architecture can be changed by adjusting the params in 'run.m' file
  2. Activation Units can be changed by specifying the model architecture in 'define_model.m' 
  3. Loss is Cross Entroy Error Function (Negative Log-Likelihood Function)
  4. Other params : Learning Rate, Momentum, Weight Decay, Dropout, Epochs
  
Results:
  
 [! Alt text](https://cloud.githubusercontent.com/assets/5204400/18943193/1a442784-85ec-11e6-8a7f-3907c4c578cb.jpg)
 
 
