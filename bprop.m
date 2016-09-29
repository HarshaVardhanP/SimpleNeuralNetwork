%------Backward Propagation------%
function model = bprop(LossGrad,model,y,t)
    nLayers = size(model.weights,2);
    %-----Output Gradient----%  
    T = zeros(10,1);
    T(t+1)= 1;
    Op = -(T-y);
    Op;
    model.gradpreacts{nLayers} = Op;
    for nL = nLayers:-1:1
        model.prevgradweights{nL} = model.gradweights{nL};
        model.prevgradbiases{nL} = model.gradbiases{nL};
        if nL > 1
            %-- Weights & Biases --%
            model.gradweights{nL} =    model.hiddens{nL-1} * model.gradpreacts{nL}' ; 
            model.gradbiases{nL} = model.gradpreacts{nL}; 
            %--
            model.gradhiddens{nL-1} =  model.weights{nL} * model.gradpreacts{nL} ;
            model.gradhiddens{nL-1} =  model.gradhiddens{nL-1}.*model.dropoutvector{nL-1};
            %size(model.gradhiddens{nL-1})
            %---For Sigmoid--%
            a = model.preacts{nL-1};
            gradActs = NN.mySigmoid(a).*(1-NN.mySigmoid(a));
            %size(gradActs)
            model.gradpreacts{nL-1} = model.gradhiddens{nL-1}.* gradActs;
        end
        if nL == 1  %---First Layer - No updates to inputs/activations
            %-- Weights & Biases --%
            model.gradweights{nL} =  (model.data) * model.gradpreacts{nL}'; 
            model.gradbiases{nL} = (model.gradpreacts{nL}); 
        end
    end 
end