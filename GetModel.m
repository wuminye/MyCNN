function [ model ] = GetModel(input)
%GETMODEL Summary of this function goes here
%   Detailed explanation goes here

num_layer = 7;

Layer = cell(num_layer,1);

Layer{1}.type = 'Input';
Layer{1}.out = input;

Layer{2}.type = 'Conv';
Layer{2}.kernelsize = [3 3];
Layer{2}.mapnum  =   6;

Layer{3}.type = 'Conv';
Layer{3}.kernelsize = [3 3];
Layer{3}.mapnum  =   16;

Layer{4}.type = 'Pooling';
Layer{4}.kernelsize = [3 3];

Layer{5}.type = 'Conv';
Layer{5}.kernelsize = [3 3];
Layer{5}.mapnum  =   24;

Layer{6}.type = 'Reshape';
Layer{6}.kernelsize = [1 1 36*24];

Layer{7}.type = 'SoftMax';
Layer{7}.out  = [10 1];


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
num_layer = 5;

Layer = cell(num_layer,1);

Layer{1}.type = 'Input';
Layer{1}.out = [input];


Layer{2}.type = 'Conv';
Layer{2}.kernelsize = [21 21];
Layer{2}.mapnum  =   1;


Layer{3}.type = 'Pooling';
Layer{3}.kernelsize = [2 2];

Layer{4}.type = 'Conv';
Layer{4}.kernelsize = [4 4];
Layer{4}.mapnum  =   2;


Layer{5}.type = 'ANN';
Layer{5}.out  = [10 1];
%}

model = InitModel(Layer);

end

