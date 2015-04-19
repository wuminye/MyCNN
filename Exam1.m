
addpath('./Core/');
addpath('./Util/');

model = InitCNNModel();

num_train = 2;
load picdata2

images = images(:,:,:,1:num_train);
labels = labels(1:num_train,:);

theta = SaveNetTheta(model);

err = gradcheck( theta ,images,labels,model,1e-6);


