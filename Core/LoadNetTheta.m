function [ model ] = LoadNetTheta( theta , old_model )
%读取参数序列到模型

model = old_model;
num_sublayer = length(old_model.sublayer); %获得有几层.

pos= 1;
for i = 2 : num_sublayer
    num_subnet = length(model.sublayer{i}.subnet); %获取当前层subnet的个数
    for j = 1 :  num_subnet
         n = length(SaveTheta(model.sublayer{i}.subnet{j}.model));  
         
         model.sublayer{i}.subnet{j}.model = ...
             LoadTheta(theta(pos:pos+n-1), model.sublayer{i}.subnet{j}.model);
         
         pos = pos+n;
    end
end


end

