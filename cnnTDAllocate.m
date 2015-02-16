function [ tX,ty,model,ind] = cnnTDAllocate(model,X,y ,pn )
%CNNTDALLOCATE Summary of this function goes here
%   ind ��¼ԭʼλ��
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

%�����������
tnn = min(ceil(pn*0.4),size(Xt,4));
index = randperm(size(Xt,4),tnn); 
tXt = Xt(:,:,:,index);
tyt = yt(index,:);
indt=indt(index);
model=cnnLog(model,'errdata: %d.\n',tnn);


[ J , cor ] = cnnAnalyze( model,size(tXt,4),tXt,tyt);
model=cnnLog(model,'Correction for errdata: %.5f%% | Cost: %e \n',cor,J);

pn = pn - tnn;
factor = rand*0.6 + 0.2;


index = randperm(size(Xp,4),ceil(pn*factor)); 
tXp = Xp(:,:,:,index);
typ = yp(index,:);
indp=indp(index);

index = randperm(size(Xf,4),pn - size(tXp,4)); 
tXf = Xf(:,:,:,index);
tyf = yf(index,:);
indf=indf(index);

tX = zeros([size(tXp,1) size(tXp,2) size(tXp,3) size(tXp,4)]+[0 0 0 size(tXf,4)+size(tXt,4)]);
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

