function [ outmodel ] = cnnTrainModel( model, X , y , step, ng ,lambda )
%CNNTRAINMODEL Summary of this function goes here
%   Detailed explanation goes here
theta = SaveTheta(model);
for i = 1: step
   [ J, grad ] = CostFunctionSGD( theta, X , y, model , lambda);
   fprintf('%e\n',mean(grad));
   theta = theta - ng*grad;
end
outmodel = model;

end

