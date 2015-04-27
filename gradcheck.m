
addpath('./Core/');
addpath('./Util/');

model = InitCNNModel();

num_train = 4;
load picdata


images = images(:,:,:,1:num_train);
labels = labels(1:num_train,:);

if strcmp(model.type, 'small') ==1
    images =  NormalizeIMG( images ,0.5);
else    
    images =  NormalizeIMG(images);
end

theta = SaveNetTheta(model);

err = NetGradCheck( theta ,images,labels,model,1e-6);

disp(mean(err));
disp(std(err));
