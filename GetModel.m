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
   
   num_layer = 6;
   Layer = cell(num_layer,1);
   
   Layer{1}.type = 'Input';
   Layer{1}.out = input;
   
   Layer{2}.type = 'Conv';
   Layer{2}.kernelsize = [3 3];
   Layer{2}.mapnum  =   8;
   
   Layer{3}.type = 'Pooling';
   Layer{3}.kernelsize = [2 2];
   
   Layer{4}.type = 'Conv';
   Layer{4}.kernelsize = [9 9];
   Layer{4}.mapnum  =   32;
   
   Layer{5}.type = 'ANN';
   Layer{5}.out = [32 1];
   
   Layer{6}.type = 'SoftMax';
   Layer{6}.out  = [2 1];
   
end


if strcmp(name,'face2')
   input = [36 32 1];
    
   num_layer = 8;
   Layer = cell(num_layer,1);

   Layer{1}.type = 'Input';
   Layer{1}.out = input;
      
   Layer{2}.type = 'Conv';
   Layer{2}.kernelsize = [5 5]; %  32 28
   Layer{2}.mapnum  =  4;
   
   Layer{3}.type = 'Pooling';
   Layer{3}.kernelsize = [2 2]; %  16 14
   
   
   Layer{4}.type = 'Conv';
   Layer{4}.kernelsize = [3 3]; %  14 12
   Layer{4}.mapnum  =   14;
      
   Layer{5}.type = 'Pooling';
   Layer{5}.kernelsize = [2 2]; %  7 6

   Layer{6}.type = 'Conv';
   Layer{6}.kernelsize = [7 6];  % 1 1
   Layer{6}.mapnum  =   40;
     

   Layer{7}.type = 'ANN';
   Layer{7}.out = [27 1];

   Layer{8}.type = 'SoftMax';
   Layer{8}.out = [2 1];
   
end
   
model = InitModel(Layer);

model.lambda = 0.00002;
model.dataname = name;  %数据库名称

model.num_train = 300; %用于训练的样本数量
model.MaxIter = 350; % 批量梯度法的迭代次数

model.testnum = 20 ; %每批训练前后 测试样本的数量
model.traintestnum = 100 ; %每多批训练前后 测试样本的数量
model.tick = 15 ; %训练时的刻度
model.itn =  30 ; %每批训练最大迭代次数
model.step = 60 ; %训练的批次数
model.interval = model.step/7;  %每隔多少批一次大检查
model.reservation = 0.00001;    %每批最少保留的样本比例
model.rate = 0.18;    %每批样本数量缩放比例
model.itreservation = 0.2;    %每批最少保留的迭代次数比例

model.logn = 0 ; %初始化日志计数
end

