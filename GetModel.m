function [ model ] = GetModel(input)
%GETMODEL Summary of this function goes here
%   Detailed explanation goes here


num_layer = 6;

Layer = cell(num_layer,1);

Layer{1}.type = 'Input';
Layer{1}.out = [input];


Layer{2}.type = 'Conv';
Layer{2}.kernelsize = [3 3];
Layer{2}.mapnum  =   2;

Layer{3}.type = 'Pooling';
Layer{3}.kernelsize = [2 2];

Layer{4}.type = 'Conv';
Layer{4}.kernelsize = [6 6];
Layer{4}.mapnum  =   4;

Layer{5}.type = 'Pooling';
Layer{5}.kernelsize = [8 8];

Layer{6}.type = 'ANN';
Layer{6}.out  = [10 1];


%{
num_layer = 3;
Layer = cell(num_layer,1);

Layer{1}.type = 'Input';
Layer{1}.out  = [input 1];

Layer{2}.type = 'ANN';
Layer{2}.out  = [81 1];

Layer{3}.type = 'ANN';
Layer{3}.out  = [10 1];
%}
%{
num_layer = 2;

Layer = cell(num_layer,1);
Layer{1}.type = 'Input';
Layer{1}.out = [input];

%Layer{2}.type = 'Pooling';
%Layer{2}.kernelsize = [28 28];

Layer{2}.type = 'ANN';
Layer{2}.out  = [10 1];

%}
model = InitModel(Layer);

end

