classdef NN
    % Modules of Neural Network
    % Includes various activation functions    
    properties
    end
    
    methods (Static)
        % Random Binary Sequence with 0.5 probability 
        function x = randBinary(n,val)
            % Total length 'n' must be even to have even # of 0s and 1s.
            nOnes = n*val;
            % List of random locations.
            indexes = randperm(n);
            x = ones(n,1);
            if nOnes == 0
                % do nothing
            else
                x(indexes(1:nOnes)) = 0;
            end
        end
        
        % Sigmoid Activation
        function y = mySigmoid(x)
            y = 1./(1+exp(-x));
        end
        
        % Softmax Activation
        % Arg : x is input matrix ; nSamples x nClasses
        function y = mySoftmax(x)
            x_sum = sum(exp(x'))';
            x_sum_repmat = repmat(x_sum,1,size(x,2));
            y = exp(x)./x_sum_repmat;
        end
        
        % Cross-Entropy-Error
        % Arg:  y is output of fprop ; Dim : nSamples x nClasses
        %       t is Target Class Labels; Dim : nSamples x 1
        function [err,grad] = myCrossEntropy(y,t)
            % -- Loss Error Calculation--%
            err = -log(y(t+1));                                                     
            %----Loss Gradient Calculation --- %
            grad = -1/(y(t+1));                            
        end        
    end
end

