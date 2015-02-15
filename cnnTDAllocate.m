function [ tX,ty,model] = cnnTDAllocate(model,X,y ,pn )
%CNNTDALLOCATE Summary of this function goes here
%   Detailed explanation goes here
num_train = size(X,4);

tp = (y(:,1)==1);
tf = (y(:,2)==1);


Xp = X(:,:,:,tp);
Xf = X(:,:,:,tf);
yp = y(tp,:);
yf = y(tf,:);

factor = rand*0.7 + 0.3;

index = randperm(size(Xp,4),ceil(pn*factor)); 
tXp = Xp(:,:,:,index);
typ = yp(index,:);


index = randperm(size(Xf,4),pn - size(tXp,4)); 
tXf = Xf(:,:,:,index);
tyf = yf(index,:);

tX = zeros(size(tXp)+[0 0 0 size(tXf,4)]);
ty = zeros(size(tX,4),size(typ,2));

tX(:,:,1,1:size(tXp,4)) = tXp;
ty(1:size(tXp,4),:) = typ;
tX(:,:,1,size(tXp,4)+1:end) = tXf;
ty(size(tXp,4)+1:end,:) = tyf;


model=cnnLog(model,'%Allocation index',index);
end

