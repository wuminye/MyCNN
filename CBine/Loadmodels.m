function [ models ] = Loadmodels(n)
%LOADMODELS Summary of this function goes here
%   Detailed explanation goes here
models = cell(n,1);
for i = 1:n
    modelname = ['./models/model',num2str(i),'.mat'];
    
    load(modelname);
    
    models{i} = model;
    
end

end

