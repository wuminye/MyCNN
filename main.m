tic;

model = GetModel('faces');

num_train = model.num_train;

[images , labels] = LoadData(model.dataname);

images = images(:,:,:,1:num_train);
labels = labels(1:num_train);


F=@(p)CostFunction( p, images, labels, model );

%FF = @(p)checkcf(p,squeeze(images)',labels(1:num_train),0.01,[784 81 10]');
theta = SaveTheta(model);
model=cnnLog(model,'Start trainingI... %s\n',datestr(now));
options = optimset('MaxIter', model.MaxIter);
[nn_params, cost,model] = fmincg(model,F, theta, options);
model = LoadTheta(nn_params,model);
save model  model
toc;