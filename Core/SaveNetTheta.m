function [ theta ] = SaveNetTheta( model )
%读取模型参数到序列

num_sublayer = length(model.sublayer); %获得有几层.
theta = [];

for i = 2 : num_sublayer
    num_subnet = length(model.sublayer{i}.subnet); %获取当前层subnet的个数
    for j = 1 :  num_subnet
        theta = [theta ; SaveTheta(model.sublayer{i}.subnet{j}.model)];       
    end
end

end

