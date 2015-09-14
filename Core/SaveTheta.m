function [ theta ] = SaveTheta( model )
%SAVETHETA Summary of this function goes here
%   Detailed explanation goes here
num = length(model.Layer);
theta = [];
for i = 2 : num
   t = model.Layer{i};
   cur = t.type;
   if strcmp(cur,'Conv') || strcmp(cur,'Convs')
       theta = [theta ; t.b(:)];
       theta = [theta ; t.w(:)];
       theta = [theta ; t.beta(:)];
   end
   
   if strcmp(cur,'ANN')
       theta = [theta ; t.b(:)];
       theta = [theta ; t.w(:)];
   end
   
    if strcmp(cur,'SoftMax')
       theta = [theta ; t.w(:)];
    end
    
    if strcmp(cur,'Pooling')
       theta = [theta ; t.b(:)];
       theta = [theta ; t.w(:)];
    end
    
end
end

