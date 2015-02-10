tic;

model = GetModel('faces');
num_train = model.num_train;

[images , labels] = LoadData(model.dataname);

images = images(:,:,:,1:num_train);
labels = labels(1:num_train,:);

%images = reshape(images,784,1,size(images,3));





model=cnnLog(model,'Start training II... %s\n',datestr(now));
step = model.step;
[ model ] = cnnTrainModel( model, images , labels , step );

save model2  model

[ J , cor ] = cnnAnalyze( model);
model=cnnLog(model,'$[ Correction: %.5f%% | Cost: %e ]$\n',cor,J);
toc;