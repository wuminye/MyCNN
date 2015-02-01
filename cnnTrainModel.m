function [ outmodel ] = cnnTrainModel( model, X , y , step, ng ,lambda )
%CNNTRAINMODEL Summary of this function goes here
%   Detailed explanation goes here
theta = SaveTheta(model);
for i = 1: step
   [ J, grad ] = CostFunction( theta, X , y, model , lambda);
   theta = theta - ng*grad;
end
outmodel = model;

end

