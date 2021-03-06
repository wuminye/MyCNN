function [ model ] = LoadTheta( theta , old_model )
%LOADTHETA Summary of this function goes here
%   Detailed explanation goes here
model = old_model;
num = length(model.Layer);
pos = 1;
for i = 2 : num
    t = model.Layer{i};
    cur = t.type;
    if strcmp(cur,'Conv') ||  strcmp(cur,'Convs')
         n = length(t.b(:));
         model.Layer{i}.b = reshape(theta(pos:pos+n-1),size(t.b));
         pos=pos+n;
         n = length(t.w(:));
         model.Layer{i}.w = reshape(theta(pos:pos+n-1),size(t.w));
         pos=pos+n;
        
    end
    
    if strcmp(cur,'Pooling')
         continue;
    end
    
    if strcmp(cur,'ANN')
         n = length(t.b(:));
         model.Layer{i}.b = reshape(theta(pos:pos+n-1),size(t.b));
         pos=pos+n;
         n = length(t.w(:));
         model.Layer{i}.w = reshape(theta(pos:pos+n-1),size(t.w));
         pos=pos+n;
        
    end
     if strcmp(cur,'SoftMax')
         n = length(t.w(:));
         model.Layer{i}.w = reshape(theta(pos:pos+n-1),size(t.w));
         pos=pos+n;   
     end
end;

end

