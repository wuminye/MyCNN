addpath('./Core/');
addpath('./Util/');
addpath('./MNIST/');
load model




if strcmp(model.type, 'small') ==1
   load picdatasmall
   % images =  NormalizeIMG( images ,0.5);
else    
    load picdata
    %images =  NormalizeIMG(images);
end


model=cnnLog(model,'Start training I... %s\n',datestr(now));
step = model.step;
[ model ] = cnnTrainModel( model, images , labels , step );

save model  model

[ J , cor ,nul,indf,model,tp,tn] = cnnAnalyze( model,size(images,4),images,labels);
model=cnnLog(model,'$[ Correction: %.5f%% | Cost: %e ]$\n',cor,J);
model=cnnLog(model,'TP: %.4f\tTN: %.4f\n',tp,tn);

