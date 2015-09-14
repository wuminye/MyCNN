function [res] = cnnPredict( model,data )
%CNNPREDICT Summary of this function goes here
%   Detailed explanation goes here


res = zeros(size(data,4),1);


for i = 1:size(data,4)
  [tt , ~] = cnnCalcForward( model ,data(:,:,1,i));
     
   [~,res(i)]=max(res{end}{end}{end}(:));
end
    
    
end

