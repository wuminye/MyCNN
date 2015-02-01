function [ outmodel ] = cnnTrainModel( model, X , y , step, ng ,lambda )
%CNNTRAINMODEL Summary of this function goes here
%   Detailed explanation goes here
theta = SaveTheta(model);
num_train = size(X,4);
%pn = ceil(num_train./150); % 随机取的样本个数
pn=24;
for i = 1: step
  fprintf('============ %d =============%d\n',i,pn);
   index = randperm(num_train,pn); 
   F=@(p)CostFunction( p, X(:,:,:,index), y(index), model ,0.01);
  %[ J, grad ] = CostFunctionSGD( theta, X , y, model , lambda);
  options = optimset('MaxIter', 30);
  [nn_params, cost] = fmincg(F, theta, options);
  theta = nn_params;
end
model = LoadTheta(theta,model);
outmodel = model;

end

