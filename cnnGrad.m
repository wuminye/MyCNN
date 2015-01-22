function [ res ] = cnnGrad( model, data , y ,m )
%CNNGRAD Summary of this function goes here
%   Detailed explanation goes here
num = length(model.Layer);
res = cell(num,1);
res{num}.b = (data{num} - y)./m;
res{num}.w = (data{num} - y)*data{num-1}'./m;
for i = num-1 :-1: 2
    t = model.Layer{i};
    cur = t.type;
    res{i}.b = 0;
    res{i}.w = 0;
    if strcmp(cur,'Conv')
        tem = t.w.^2;
        res = res + sum(tem(:));
    end
    
    if strcmp(cur,'ANN')
        res{i}.b = model.Layer{i+1}.w'*res{i+1}.b.*(data{i}.*(1-data{i}));
        res{i}.w = res{i}.b*data{i-1}';
    end
end

end

