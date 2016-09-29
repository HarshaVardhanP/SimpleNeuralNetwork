%--------Update Parameters with Gradients---%
function model = updateParams(model,lr,mu)
     nLayers = size(model.weights,2);
     for i=1:nLayers
         model.weights{i} = model.weights{i}-lr*(model.gradweights{i})-lr*mu*(model.prevgradweights{i}); %(0.001*model.weights{i});
         model.biases{i} = model.biases{i}-lr*(model.gradbiases{i})-lr*mu*(model.prevgradbiases{i});         
     end
end