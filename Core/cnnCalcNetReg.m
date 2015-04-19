function [ res ] = cnnCalcNetReg( model )
%计算网络正则项

res = 0;
num_sublayer = length(model.sublayer); %获得有几层.

for i = 2 : num_sublayer
    num_subnet = length(model.sublayer{i}.subnet); %获取当前层subnet的个数
    for j = 1 :  num_subnet
        res = res + cnnCalcReg(model.sublayer{i}.subnet{j}.model);
    end
end

end

