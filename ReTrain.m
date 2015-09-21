addpath('./MNIST/');
addpath('./Core/');
addpath('./Util/');

pn = input('the number of training : ');
itn = input('the number of iteration : ');
dropout = input('Enable Dropout? : ');
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
  if dropout == 1
   model = DropoutStart(model);
  end
  [nn_params, cost , model ,corind] = fmincg(model,F, theta, options);
  if dropout == 1
     model = DropoutEnd(model);
  end
  model = LoadNetTheta(nn_params,model);
  
  
  save model  model
toc;