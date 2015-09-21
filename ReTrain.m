addpath('./MNIST/');
addpath('./Core/');
addpath('./Util/');

pn = input('the number of training : ');
itn = input('the number of iteration : ');

load model


    load picdata;

theta = SaveNetTheta(model);
model=cnnLog(model,'ReTrain\n');
tic;
  
   

  %分配每批训练样本
  [ tX,ty,model,ind] = cnnTDAllocate(model,images,labels ,pn );
  model=cnnLog(model,'faces:%d\n',sum(ty(:,1)==1));
  options = optimset('MaxIter', itn);
  F=@(p)CostFunction( p, tX, ty, model );
   model = DropoutStart(model);
  [nn_params, cost , model ,corind] = fmincg(model,F, theta, options);
  model = DropoutEnd(model);
  model = LoadNetTheta(nn_params,model);
  
  
  save model  model
toc;