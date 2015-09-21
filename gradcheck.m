addpath('./MNIST/');
addpath('./Core/');
addpath('./Util/');

model = InitCNNModel();

num_train = 4;
load picdata


images = images(:,:,:,1:num_train);
labels = labels(1:num_train,:);



theta = SaveNetTheta(model);
 model = DropoutStart(model);
err = NetGradCheck( theta ,images,labels,model,1e-6);
model = DropoutEnd(model);
disp(mean(err));
disp(std(err));
