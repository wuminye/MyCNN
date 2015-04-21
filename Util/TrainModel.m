addpath('./Core/');
addpath('./Util/');

load model


load picdata


model=cnnLog(model,'Start training I... %s\n',datestr(now));
step = model.step;
[ model ] = cnnTrainModel( model, images , labels , step );

save model  model

[ J , cor ,nul,indf] = cnnAnalyze( model,size(images,4),images,labels);
model=cnnLog(model,'$[ Correction: %.5f%% | Cost: %e ]$\n',cor,J);


