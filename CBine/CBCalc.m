function [ res ] = CBCalc( models,CBmodel,data)
%CBCALC Summary of this function goes here
%   Detailed explanation goes here
MN = length(models);
features = [];

for i = 1 : MN
    
   res = cnnCalcForward(models{i},data);
   output = res{end}{end}{end-1};
   features = [features ; output(:)];
end





end

