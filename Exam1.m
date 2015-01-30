
num_train = 5;

imageDim = 28;
images = loadMNISTImages('train-images.idx3-ubyte');
images = reshape(images,imageDim,imageDim,1,[]);
images = images(:,:,:,1:num_train);
disp(size(images));
labels = loadMNISTLabels('train-labels.idx1-ubyte');
labels(labels==0) =10;

%images = reshape(images,784,1,size(images,3));
model = GetModel([28 28 1]);

F=@(p)CostFunction( p, images , labels(1:num_train), model ,0.01);
FF = @(p)checkcf(p,squeeze(images)',labels(1:num_train),0.01,[784 81 10]');
theta = SaveTheta(model);
fprintf('Start Checking ....\n');

err = gradcheck( theta ,images,labels(1:num_train),model,0.01,1e-6);
%save nn_params