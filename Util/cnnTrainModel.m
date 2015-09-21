function [ outmodel ] = cnnTrainModel( model, X , y , step )


theta = SaveNetTheta(model);
%num_train = size(X,4);
%pn = ceil(num_train./150); % 随机取的样本个数
images = X;
labels = y;
num_train = model.num_train;
index = randperm(size(images,4),num_train); 
X = images(:,:,:,index);
y = labels(index,:);

model=cnnLog(model,'%Train_sample index',index);
%对错表
model.corind = ones(num_train,1);

for i = 1: step
  if mod(i,floor(model.interval))==0
      [ J , cor,nul ,indf,model] = cnnAnalyze( model,model.traintestnum,images,labels);
        
      model=cnnLog(model,'\n*[ Correction: %.5f%% | Cost: %e ]*\n\n',cor,J);
  end
  [pn,itn] = getpn(model,i,step,num_train);
  model=cnnLog(model,'< %d > Num_train: %d  Iter_num: %d \n',i,pn,itn);
  
  [ J , cor ,nul,indf,model] = cnnAnalyze( model,model.testnum,images,labels);
  model=cnnLog(model,'[ Correction: %.5f%% | Cost: %e ]\n',cor,J);
  precor = cor;  %前一次的正确率
  trainflag = 1;  %判断是否要用当前样本继续训练
  
  %分配每批训练样本
  [ tX,ty,model,ind] = cnnTDAllocate(model,X,y ,pn );
 
 
  model=cnnLog(model,'faces:%d\n',sum(ty(:,1)==1));
  [ J , cor ,nul ,indf,model] = cnnAnalyze( model,size(tX,4),tX,ty);
  model=cnnLog(model,'Correction for train: %.5f%% | Cost: %e \n',cor,J);
 

  
  cost = [];
  while trainflag == 1
      theta = SaveNetTheta(model);
      F=@(p)CostFunction( p, tX, ty, model );
      options = optimset('MaxIter', itn);
      model = DropoutStart(model);
      [nn_params, cost1 , model ,corind] = fmincg(model,F, theta, options);
      model = DropoutEnd(model);
      cost = [cost;cost1];
      %{
      %更新错误记录
      model.corind(ind(logical(corind))) = 1; 
      model.corind(ind(~logical(corind))) = 0; 
%}
     
      %保存结果至model.mat
      ttt = model;
      model = LoadNetTheta(nn_params,model);
      theta = nn_params;
     % ShowLayer( model, X(:,:,1,1) ,y(1,:) );
      %saveas(gcf,'data.fig');

      save model  model

      [ J , cor ,ind ,indf,model,tp,tn] = cnnAnalyze( model,model.testnum,X,y);

      %更新错误表
      model.corind(ind) = 1; 
      model.corind(indf) = 0; 
  
      if abs(precor - cor) < 2 || precor>cor
          %model = ttt; %还原模型
          model=cnnLog(model,'===Bad Samples==  %.5f\n',cor);
          trainflag = 0;
      else 
          precor = cor;
      end
      
  end
   model=cnnLog(model,'%fmincg result\n',cost);
  model=cnnLog(model,'[ Correction: %.5f%% | Cost: %e ]\n',cor,J);
  model=cnnLog(model,'TP: %.4f\tTN: %.4f\n',tp,tn);
  model=cnnLog(model,'------ Cost: %e | %.5f%% -----\n\n',cost(end),cost(1)/cost(end)*100);
 %{ 
 [ J , cor,nul ,indf,model] = cnnAnalyze( model,model.traintestnum,images,labels);
 model=cnnLog(model,'\n*[ Correction: %.5f%% | Cost: %e ]*\n\n',cor,J);
 
 if cor > 96
     model=cnnLog(model,'Goals achieved. quit.\n');
     break;
 end
  %}
%=============================================================
% 调整lambda 
  ppp = cost(1)/cost(end)*100;
  
  if ppp>1000
      model.lambda = model.lambda*1.1;
      model=cnnLog(model,'[lambda changed]: %e \n',model.lambda);
  end
  if ppp<150
      model.lambda = model.lambda*0.8;
      model=cnnLog(model,'[lambda changed]: %e \n',model.lambda);
  end
%========================================================  



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
   
   
    cur = ceil( a*i^2+b); 
   %计算迭代次数
   itn = ceil((1-model.itreservation)*itn*(1 - cur/tick)+model.itreservation*itn);
   if step == 1
      pn = 1;
      itn=1;
   end
end