tic;

model = GetModel('face2');

%[images , labels] = LoadData(model.dataname);
load picdata


%images = reshape(images,784,1,size(images,3));





model=cnnLog(model,'Start training II... %s\n',datestr(now));
step = model.step;
[ model ] = cnnTrainModel( model, images , labels , step );

save model2  model

[ J , cor ,nul,indf] = cnnAnalyze( model,size(images,4),images,labels);
model=cnnLog(model,'$[ Correction: %.5f%% | Cost: %e ]$\n',cor,J);
toc;