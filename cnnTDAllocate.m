function [ tX,ty,model,ind] = cnnTDAllocate(model,X,y ,pn )
%CNNTDALLOCATE Summary of this function goes here
%   ind 记录原始位置
num_train = size(X,4);
indt = [1:num_train]';
indp = [1:num_train]';
indf = [1:num_train]';

tt = (model.corind==0);
tp = (y(:,1)==1);
tf = (y(:,2)==1);

indt=indt(tt);
indp=indp(tp);
indf=indf(tf);

Xt = X(:,:,:,tt);
Xp = X(:,:,:,tp);
Xf = X(:,:,:,tf);
yt = y(tt,:);
yp = y(tp,:);
yf = y(tf,:);

%分配错误样本
tnn = min(ceil(pn*0.4),size(Xt,4));
index = randperm(size(Xp,4),tnn); 
tXt = Xt(:,:,:,index);
tyt = yt(index,:);
indt=indt(index);
fprintf('errdata: %d.\n',tnn);

pn = pn - tnn;
factor = rand*0.8 + 0.2;


index = randperm(size(Xp,4),ceil(pn*factor)); 
tXp = Xp(:,:,:,index);
typ = yp(index,:);
indp=indp(index);

index = randperm(size(Xf,4),pn - size(tXp,4)); 
tXf = Xf(:,:,:,index);
tyf = yf(index,:);
indf=indf(index);

tX = zeros(size(tXp)+[0 0 0 size(tXf,4)+size(tXt,4)]);
ty = zeros(size(tX,4),size(typ,2));
ind = zeros(size(tX,4),1);

tX(:,:,1,1:size(tXt,4)) = tXt;
ty(1:size(tXt,4),:) = tyt;
tX(:,:,1,size(tXt,4)+1:size(tXt,4)+size(tXp,4)) = tXp;
ty(size(tXt,4)+1:size(tXt,4)+size(tXp,4),:) = typ;
tX(:,:,1,size(tXt,4)+size(tXp,4)+1:end) = tXf;
ty(size(tXt,4)+size(tXp,4)+1:end,:) = tyf;
ind(1:size(tXt,4)) = indt;
ind(size(tXt,4)+1:size(tXt,4)+size(tXp,4)) = indp;
ind(size(tXt,4)+size(tXp,4)+1:end) =indf;

model=cnnLog(model,'%Allocation index',index);
end

