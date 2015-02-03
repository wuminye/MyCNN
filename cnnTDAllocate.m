function [ tX,ty,model] = cnnTDAllocate(model,X,y ,pn )
%CNNTDALLOCATE Summary of this function goes here
%   Detailed explanation goes here
num_train = size(X,4);
index = randperm(num_train,pn); 
tX = X(:,:,:,index);
ty = y(index);

model=cnnLog(model,'%Allocation index',index);
end

