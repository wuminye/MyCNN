function [ outmodel ] = cnnTrainModel( model, X , y , step )
%CNNTRAINMODEL Summary of this function goes here
%   Detailed explanation goes here
theta = SaveTheta(model);
num_train = size(X,4);
%pn = ceil(num_train./150); % 随机取的样本个数
images = X;
labels = y;
num_train = model.num_train;
X = images(:,:,:,1:num_train);
y = labels(1:num_train,:);


for i = 1: step
  if mod(i,floor(model.interval))==0
      [ J , cor ] = cnnAnalyze( model,model.traintestnum,images,labels);
      model=cnnLog(model,'\n*[ Correction: %.5f%% | Cost: %e ]*\n\n',cor,J);
  end
  [pn,itn] = getpn(model,i,step,num_train);
  model=cnnLog(model,'< %d > Num_train: %d  Iter_num: %d \n',i,pn,itn);
  
  [ J , cor ] = cnnAnalyze( model,model.testnum,images,labels);
  model=cnnLog(model,'[ Correction: %.5f%% | Cost: %e ]\n',cor,J);
  
  %分配每批训练样本
  [ tX,ty,model] = cnnTDAllocate(model,X,y ,pn );
  
  fprintf('faces:%d\n',sum(ty(:,1)==1));
  
  
  F=@(p)CostFunction( p, tX, ty, model );
  options = optimset('MaxIter', itn);
  [nn_params, cost , model] = fmincg(model,F, theta, options);
  model=cnnLog(model,'%fmincg result\n',cost);
  %保存结果至model2.mat
  model = LoadTheta(nn_params,model);
  
  ShowLayer( model, X(:,:,1,1) ,y(1,:) );
  saveas(gcf,'data.fig');
  
  save model2  model
  
  [ J , cor ] = cnnAnalyze( model,model.testnum,images,labels);
  model=cnnLog(model,'[ Correction: %.5f%% | Cost: %e ]\n',cor,J);
  model=cnnLog(model,'------ Cost: %e | %.5f%% -----\n\n',cost(end),cost(1)/cost(end)*100);
  
  theta = nn_params;
end

outmodel = model;

end

function [pn ,itn] = getpn(model,i,step,num)
   tick = model.tick ;%刻度    
   itn  = model.itn ;%最大迭代次数
   
   tem = floor(step/tick);
     
   %计算需要处理的批次数
   %{
   a = 9/log(step);
   k = exp(1/a);
   cur = 10 - ceil(a*log((step-i+1)*k));
   %}
   %二次函数模型
   a = (1-tick)/(step^2-1);
   b = tick - a;
   cur = ceil(tick - a*i^2-b);
   
   
   %计算样本数量
   y = (10/step)^2*( (cur*tem)^2 + ((cur-1)*tem)^2 )/2;
   y = y * model.rate;
   pn = ceil(y/100*num*(1-model.reservation) + model.reservation*num);
   
   %计算迭代次数
   itn = ceil((1-model.itreservation)*itn*(1 - cur/tick)+model.itreservation*itn);
   if step == 1
      pn = 1;
      itn=1;
   end
end