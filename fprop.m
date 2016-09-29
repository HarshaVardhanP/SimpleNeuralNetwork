%--------------F-Prop-------------------%
%----phase = 1 for training and phase = 0 for testing----% 
function [Y,model] = fprop(data,model,phase)
    nLayers = size(model.weights,2);
    model.data = data(:,1:end-1)';
    A = model.data;
    for i = 1:nLayers
       A = [A;1];
       size(A);
       % Linear %
       W = [model.weights{i} ; model.biases{i}'];
       A = W'*A;
       model.preacts{i} = A;
       if i < nLayers
        % Sigmoid Activation %
        A = NN.mySigmoid(A);
        if phase == 1
            model.dropoutvector{i} = NN.randBinary(size(model.dropoutvector{i},1),model.dropout_val); 
            A = A.*model.dropoutvector{i};
        else
            A = A.*ones(size(model.dropoutvector{i},1),1)*0.5; 
        end
        model.hiddens{i} = A;
       end
    end
    % Softmax 
    Y = NN.mySoftmax(A);
end