function [ res ] = cnnCalcReg( model )
%CNNCALCREG Summary of this function goes here
%   Detailed explanation goes here
num = length(model.Layer);
res = 0;
for i = 2 : num
    t = model.Layer{i};
    cur = t.type;
    if strcmp(cur,'Conv') || strcmp(cur,'Convs')
        tem = t.w.^2;
        res = res + sum(tem(:));
    end
    
    if strcmp(cur,'ANN') || strcmp(cur,'SoftMax')
        tem = t.w.^2;
        res = res + sum(tem(:));
    end
    
end

end

