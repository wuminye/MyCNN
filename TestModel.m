addpath('./MNIST/');
addpath('./Core/');
addpath('./Util/');

load model
load testdata



 
model=cnnLog(model,'---------TestModel-------------\n');
model=cnnLog(model,'TestNum:%d\n',size(images,4));
[ J , cor,nul ,indf,model] = cnnAnalyze( model,size(images,4),images,labels);
model=cnnLog(model,'\n*[ Correction: %.5f%% | Cost: %e ]*\n\n',cor,J);
model=cnnLog(model,'--------------------------------\n');