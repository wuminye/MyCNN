function [ tX,ty ] = cnnTDAllocate(model,X,y ,pn )
%CNNTDALLOCATE Summary of this function goes here
%   Detailed explanation goes here
num_train = size(X,4);
index = randperm(num_train,pn); 
tX = X(:,:,:,index);
ty = y(index);

[ J , cor ] = cnnAnalyze( model,model.testnum);
 fprintf('[ Correction: %.5f%% | Cost: %e ]\n',cor,J);
end

