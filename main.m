tic;
num_train = 20;


[images , labels] = LoadData('MNIST');

images = images(:,:,:,1:num_train);
labels = labels(1:num_train);


model = GetModel([28 28 1]);



F=@(p)CostFunction( p, images, labels, model ,0.01);
FSGD=@(p)CostFunctionSGD( p, images, labels, model ,0.01);
%FF = @(p)checkcf(p,squeeze(images)',labels(1:num_train),0.01,[784 81 10]');
theta = SaveTheta(model);
fprintf('Start training....\n');
options = optimset('MaxIter', 400);
[nn_params, cost] = fmincg(F, theta, options);
model = LoadTheta(nn_params,model);
save model  model
toc;