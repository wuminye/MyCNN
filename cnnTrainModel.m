function [ outmodel ] = cnnTrainModel( model, X , y , step, lambda )
%CNNTRAINMODEL Summary of this function goes here
%   Detailed explanation goes here
theta = SaveTheta(model);
num_train = size(X,4);
%pn = ceil(num_train./150); % 随机取的样本个数
pn=24;
for i = 1: step
  if mod(i,50) == 0
     pn = pn*2; 
     if pn>num_train
         pn =num_train;
     end
     [ J , cor ] = cnnAnalyze( model,2000);
     fprintf('Correction: %.5f%% | Cost: %e\n',cor,J);
  end
  fprintf('============ %d =============%d\n',i,pn);
   index = randperm(num_train,pn); 
   F=@(p)CostFunction( p, X(:,:,:,index), y(index), model ,lambda);
  %[ J, grad ] = CostFunctionSGD( theta, X , y, model , lambda);
  options = optimset('MaxIter', 30);
  [nn_params, cost] = fmincg(F, theta, options);
  theta = nn_params;
end
model = LoadTheta(theta,model);
outmodel = model;

end

