function [ model ] = GetModel(name)
%GETMODEL Summary of this function goes here
%   Detailed explanation goes here

if strcmp(name,'MNIST')
   input = [28 28 1];
    
   num_layer = 8;
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

   Layer{7}.type = 'ANN';
   Layer{7}.out = [36 1];

   Layer{8}.type = 'SoftMax';
   Layer{8}.out  = [10 1];

end
   
if strcmp(name,'faces')
   input = [20 20 1];
   
   num_layer = 7;
   Layer = cell(num_layer,1);
   
   Layer{1}.type = 'Input';
   Layer{1}.out = input;
   
   Layer{2}.type = 'Conv';
   Layer{2}.kernelsize = [3 3];
   Layer{2}.mapnum  =   8;
   
   Layer{3}.type = 'Pooling';
   Layer{3}.kernelsize = [2 2];
   
   Layer{4}.type = 'Conv';
   Layer{4}.kernelsize = [3 3];
   Layer{4}.mapnum  =   24;
   
   Layer{5}.type = 'Conv';
   Layer{5}.kernelsize = [7 7];
   Layer{5}.mapnum  =   68;
   
   Layer{6}.type = 'ANN';
   Layer{6}.out = [32 1];
   
   Layer{7}.type = 'SoftMax';
   Layer{7}.out  = [2 1];
   
end
   
model = InitModel(Layer);

model.lambda = 0.05;
model.dataname = name;  %数据库名称
model.testnum = 200 ; %每批训练前后 测试样本的数量
model.traintestnum = 3000 ; %每多批训练前后 测试样本的数量
model.tick = 15 ; %训练时的刻度
model.itn =  30 ; %每批训练最大迭代次数
end

