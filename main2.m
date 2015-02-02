tic;
num_train = 10000;

[images , labels] = LoadData('MNIST');

images = images(:,:,:,1:num_train);
labels = labels(1:num_train);

%images = reshape(images,784,1,size(images,3));
model = GetModel([28 28 1]);




fprintf('Start training....\n');
step = 200;
[ model ] = cnnTrainModel( model, images , labels , step, 0.01 );

save model2  model

[ J , cor ] = cnnAnalyze( model);
fprintf('\n[ Correction: %.5f%% | Cost: %e ]\n',cor,J);
toc;