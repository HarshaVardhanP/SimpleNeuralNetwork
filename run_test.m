
%----------Load Testing Data-----------------%
[parentdir,~,~]=fileparts(pwd);
global testdata
[testdata] = textread(strcat(parentdir,'/Data/digitstest.txt'),'','delimiter',',');

%--------Load Model----%
myNN = load('model_1HN.mat');

%--------Do Forward Prop-----%
[NLLerr, Cerr, OPs] = run_valid(testdata,myNN.model);
NLLerr
Cerr