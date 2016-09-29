%----------Define Network Architecture--------%
%----inputs : layer_arr = [nInputs,nHidden_1,...nHidden_i,...nOutputs]---%
%-------------Example = [784,100,10]--------%
%-------------Linear,Sigmoid, Linear, Sigmoid, Softmax ----------- %
function model = define_model(layer_arr,dropout,batchsize)
    model.data={};
    model.dropout_val = dropout;
    
    model.weights = {};
    model.gradweights = {};
    model.prevgradweights = {}; %--for momentum
    
    model.biases = {};
    model.gradbiases = {};
    model.prevgradbiases = {}; %---for momentum
    
    model.preacts = {};
    model.gradpreacts = {};
    model.dropoutvector = {};
    
    model.hiddens = {};
    model.gradhiddens = {};
    
    for i = 1:size(layer_arr,2)-1
       init_val = sqrt(6)/sqrt(layer_arr(i)+layer_arr(i+1));
       init_val = 0.01;
       %-----Params----%
       model.weights{i} = (rand(layer_arr(i),layer_arr(i+1))-0.5)*2*init_val;  % Uniform random distribution in [-0.1,0.1]
       model.biases{i} = (rand(layer_arr(i+1),1)-0.5)*2*init_val;
       model.preacts{i} = zeros(layer_arr(i+1),1);
       
       %-----Gradients-----%
       model.gradweights{i} = zeros(layer_arr(i),layer_arr(i+1));
       model.gradbiases{i} = zeros(layer_arr(i+1),1);
       model.prevgradweights{i} = zeros(layer_arr(i),layer_arr(i+1));
       model.prevgradbiases{i} = zeros(layer_arr(i+1),1);
       
       model.gradpreacts{i} = zeros(layer_arr(i+1),1);
    end
    for i = 1:size(layer_arr,2)-2
        model.hiddens{i} = zeros(layer_arr(i+1),1);
        model.gradhiddens{i} = zeros(layer_arr(i+1),1);
        if model.dropout_val > 0
            model.dropoutvector{i} = NN.randBinary(layer_arr(i+1),model.dropout_val);
        else
            model.dropoutvector{i} = ones(layer_arr(i+1),1);
        end
    end
end