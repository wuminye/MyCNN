function [ outmodel ] = cnnTrainModel( model, X , y , step, lambda )
%CNNTRAINMODEL Summary of this function goes here
%   Detailed explanation goes here
theta = SaveTheta(model);
num_train = size(X,4);
%pn = ceil(num_train./150); % ���ȡ����������

for i = 1: step
  if mod(i,floor(step/7))==0
      [ J , cor ] = cnnAnalyze( model,3000);
      fprintf('\n*[ Correction: %.5f%% | Cost: %e ]*\n\n',cor,J);
  end
  [pn,itn] = getpn(i,step,num_train);
  fprintf('< %d > Num_train: %d  Iter_num: %d \n',i,pn,itn);
  [ tX,ty ] = cnnTDAllocate(model,X,y ,pn );
  
  F=@(p)CostFunction( p, tX, ty, model ,lambda);
  options = optimset('MaxIter', itn);
  [nn_params, cost] = fmincg(F, theta, options);
  
  %��������model2.mat
  model = LoadTheta(nn_params,model);
  save model2  model
  
  [ J , cor ] = cnnAnalyze( model,200);
  fprintf('[ Correction: %.5f%% | Cost: %e ]\n',cor,J);
  fprintf('------ Cost: %e | %.5f%% -----\n\n',cost(end),cost(1)/cost(end)*100);
  
  theta = nn_params;
end

outmodel = model;

end

function [pn ,itn] = getpn(i,step,num)
   tick = 15 ;%�̶�    
   itn  = 30 ;%����������
   
   tem = floor(step/tick);
     
   %������Ҫ�����������
   %{
   a = 9/log(step);
   k = exp(1/a);
   cur = 10 - ceil(a*log((step-i+1)*k));
   %}
   %���κ���ģ��
   a = (1-tick)/(step^2-1);
   b = tick - a;
   cur = ceil(tick - a*i^2-b);
   
   
   %������������
   y = (10/step)^2*( (cur*tem)^2 + ((cur-1)*tem)^2 )/2;
   y = y * 0.42;
   pn = ceil(y/100*num);
   
   %�����������
   itn = ceil(0.8*itn*(1 - cur/tick)+0.2*itn);
end