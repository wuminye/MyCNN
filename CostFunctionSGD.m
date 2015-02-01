function [ J, grad ] = CostFunctionSGD( theta , input , y, model ,lambda)
%COSTFUNCTIONSGD Summary of this function goes here
%   Detailed explanation goes here
num_train = size(input,4);
pn = ceil(num_train./3); % 随机取的样本个数

index = randperm(num_train,pn);

[ J, grad ] = CostFunction( theta , input(:,:,:,index) , y(index), model ,lambda);
end

