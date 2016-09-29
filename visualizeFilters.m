%-----Visualize Weights/Filters-----%


%--loading the model---%
%---Comment these two lines if model is in environment--%
myNN = load('model_1HN.mat');
model = myNN.model;

W = model.weights{1};
min_w = 0; %min(min(W));
max_w = 1; %max(max(W));
F = zeros(29,29,1,100);
for i = 1:size(W,2)
    F(1:28,1:28,:,i) = vec2mat(W(:,i),28);
    F(29,:,1,i) = ones(1,29);
    F(:,29,1,i) = ones(29,1);
end
F = F-min_w;
F = F./(-min_w+max_w);
figure,
montage(F)
