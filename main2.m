tic;
num_train = 400;

imageDim = 28;
images = loadMNISTImages('train-images.idx3-ubyte');
images = reshape(images,imageDim,imageDim,1,[]);
images = images(:,:,:,1:num_train);
disp(size(images));
labels = loadMNISTLabels('train-labels.idx1-ubyte');
labels(labels==0) =10;

%images = reshape(images,784,1,size(images,3));
model = GetModel([28 28 1]);




fprintf('Start training....\n');
step = 200;
[ model ] = cnnTrainModel( model, images , labels(1:num_train) , step, 0.4 , 0.01 );

save model  model
toc;