tic;
num_train = 5000;

model = GetModel('faces');

[images , labels] = LoadData(model.dataname);

images = images(:,:,:,1:num_train);
labels = labels(1:num_train);

%images = reshape(images,784,1,size(images,3));





fprintf('Start training....\n');
step = 50;
[ model ] = cnnTrainModel( model, images , labels , step );

save model2  model

[ J , cor ] = cnnAnalyze( model);
fprintf('\n[ Correction: %.5f%% | Cost: %e ]\n',cor,J);
toc;