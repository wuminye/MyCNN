function [ tX,ty,model,ind] = cnnTDAllocate(model,X,y ,pn )
%CNNTDALLOCATE Summary of this function goes here
%   ind ��¼ԭʼλ��

ind =0;
index = randperm(size(X,4),pn);

tX = X(:,:,:,index);
ty = y(index,:);



end

