%------Validation Data - FPROP------%
function [NLLerr, Cerr, OPs] = run_valid(validdata,model)
    nVSamples = size(validdata,1);
    res = -1*ones(nVSamples,1);
    acc = 0;
    NLLerr = 0;
    OPs = [];
    test_phase = 0;
    for j = 1:nVSamples
       [Y,model] = fprop(validdata(j,:),model,test_phase);
       OPs = [OPs Y];
       target = validdata(j,end);
       [val,idx] = max(Y);
       res(j) = idx-1;
       if res(j) == target
           acc = acc+1;
           %disp('Hi')
        end
        %----Loss Function : Cross Entropy Error----%
        [Error,LossGrad] = NN.myCrossEntropy(Y,target);
        NLLerr = NLLerr+Error;      
    end
    Cerr = 1-(acc)/nVSamples;
    Cerr = Cerr * 100;
    NLLerr = NLLerr/nVSamples;
end

