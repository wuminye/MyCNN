
num_train = 5;

model = GetModel('faces');

[images , labels] = LoadData(model.dataname);

images = images(:,:,:,1:num_train);
labels = labels(1:num_train);


%images = reshape(images,784,1,size(images,3));



F=@(p)CostFunction( p, images , labels, model );
%FF = @(p)checkcf(p,squeeze(images)',labels(1:num_train),0.01,[784 81 10]');
theta = SaveTheta(model);
fprintf('Start Checking ....\n');

err = gradcheck( theta ,images,labels,model,1e-6);
%save nn_params
