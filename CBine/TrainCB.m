function [ model ] = TrainCB( model, X, y ,N,itn)


tp = (y(:,1)==1);
tf = (y(:,2)==1);

Xp = X(:,:,:,tp);
Xf = X(:,:,:,tf);
yp = y(tp,:);
yf = y(tf,:);

index = randperm(size(Xp,4),ceil(N*0.5));
tXp = Xp(:,:,:,index);
typ = yp(index,:);

index = randperm(size(Xf,4),ceil(N*0.5)); 
tXf = Xf(:,:,:,index);
tyf = yf(index,:);

tX = zeros([size(tXp,1) size(tXp,2) size(tXp,3) size(tXp,4)]+[0 0 0 size(tXf,4)]);
ty = zeros(size(tX,4),size(typ,2));

tX(:,:,:,1:size(tXp,4)) = tXp;
ty(1:size(tXp,4),:) = typ;
tX(:,:,:,size(tXp,4)+1:end) = tXf;
ty(size(tXp,4)+1:end,:) = tyf;

model=cnnLog(model,'ReTrain\n');
theta = SaveNetTheta(model);
model=cnnLog(model,'faces:%d\n',sum(ty(:,1)==1));

options = optimset('MaxIter', itn);
F=@(p)CostFunction( p, tX, ty, model );
[nn_params, cost , model ,corind] = fmincg(model,F, theta, options);
model = LoadNetTheta(nn_params,model);



end

