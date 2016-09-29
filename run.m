clc;
clear all;
close all;
%----------Load Training Data-----------------%
[parentdir,~,~]=fileparts(pwd);
global traindata
[traindata] = textread(strcat(parentdir,'/Data/digitstrain.txt'),'','delimiter',',');
nSamples = size(traindata,1);


%----------Load Validation Data-----------------%
[parentdir,~,~]=fileparts(pwd);
global validdata
[validdata] = textread(strcat(parentdir,'/Data/digitsvalid.txt'),'','delimiter',',');
nVSamples = size(validdata,1);

%----preprocess----%
data_mean = mean(mean(traindata(:,1:end-1)));
data_std = std(std(traindata(:,1:end-1)));
traindata(:,1:end-1) = (traindata(:,1:end-1)); %-data_mean)/data_std;
validdata(:,1:end-1) = (validdata(:,1:end-1)); %-data_mean)/data_std;
%---shuffle the data-----%
traindata = traindata(randperm(size(traindata,1)),:);

%---Model Definition-----%
NN_arr  = [784,500,10];
dropout = 0.5;
lr = 0.1;
mu = 0.9;
epochs = 50;
batchsize = 1; 
global model;
model = define_model(NN_arr,dropout,batchsize);
disp(model)
%-------------------------------%


train_NLLerr = zeros(epochs,1);
train_Cerr = zeros(epochs,1);
valid_NLLerr = zeros(epochs,1);
valid_Cerr = zeros(epochs,1);

%---------Train the model----------%
% phase = 1 for Training, phase = 0 for Testing;
train_phase = 1; % always 1
test_phase = 0; % always 0
for i=1:epochs
    res = -1*ones(nSamples,1);
    Ops = [];
    for j = nSamples:-1:1
        %----Forward Prop-------%
        [Y,model] = fprop(traindata(j,:),model,train_phase);
        Ops = [Ops Y];
        target = traindata(j,end);
        [val,idx] = max(Y);
        res(j) = idx-1;
        if res(j) == target
           train_Cerr(i) = train_Cerr(i)+1;
        end
        %----Loss Function : Cross Entropy Error----%
        [Error,LossGrad] = NN.myCrossEntropy(Y,target);
        train_NLLerr(i) = train_NLLerr(i)+Error;
        if i > 0
            %----Backward Prop------%
            model = bprop(LossGrad,model,Y,target);
            %----Update Weights and Biases-----%
            model = updateParams(model,lr,mu);
        end
    end    
    train_NLLerr(i) = train_NLLerr(i)/nSamples;
    train_Cerr(i) = (1-train_Cerr(i)/nSamples)*100;
    %disp(acc)
    %------------VALIDATION---------------%
    [valid_NLLerr(i), valid_Cerr(i),OPs] = run_valid(validdata,model); 
end
figure, 
subplot(1,2,1),plot(train_NLLerr), hold on
subplot(1,2,1),plot(valid_NLLerr) 
legend('Train','Valid')
title('Cross Entropy Error')

subplot(1,2,2),plot(train_Cerr), hold on
subplot(1,2,2),plot(valid_Cerr)
legend('Train','Valid')
title('Classification Error')
%figure, plot(res)
