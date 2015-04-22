addpath('./Core/');
addpath('./Util/');

load model


load picdata


model=cnnLog(model,'Start training I... %s\n',datestr(now));
step = model.step;
[ model ] = cnnTrainModel( model, images , labels , step );

save model  model

[ J , cor ,nul,indf,model,tp,tn] = cnnAnalyze( model,size(images,4),images,labels);
model=cnnLog(model,'$[ Correction: %.5f%% | Cost: %e ]$\n',cor,J);
model=cnnLog(model,'TP: %.4f\tTN: %.4f\n',tp,tn);

