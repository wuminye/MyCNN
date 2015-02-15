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
    
   num_layer = 10;
   Layer = cell(num_layer,1);

   Layer{1}.type = 'Input';
   Layer{1}.out = input;
   
   Layer{2}.type = 'Conv';
   Layer{2}.kernelsize = [7 7]; %  30 26
   Layer{2}.mapnum  =  2;
   
   Layer{3}.type = 'Conv';
   Layer{3}.kernelsize = [3 3]; %  28 24
   Layer{3}.mapnum  =  5;
   
   Layer{4}.type = 'Pooling';
   Layer{4}.kernelsize = [2 2]; %  14 12
   
   Layer{5}.type = 'Conv';
   Layer{5}.kernelsize = [7 7]; %  8 6
   Layer{5}.mapnum  =   8;
   
   Layer{6}.type = 'Conv';
   Layer{6}.kernelsize = [3 3]; %  6 4
   Layer{6}.mapnum  =   10;
      
   Layer{7}.type = 'Pooling';
   Layer{7}.kernelsize = [2 2]; %  3 2

   Layer{8}.type = 'Conv';
   Layer{8}.kernelsize = [3 2];  % 1 1
   Layer{8}.mapnum  =   25;
     

   Layer{9}.type = 'ANN';
   Layer{9}.out = [20 1];

   Layer{10}.type = 'SoftMax';
   Layer{10}.out = [2 1];
   
end
   
model = InitModel(Layer);

model.lambda = 0.000002;
model.dataname = name;  %���ݿ�����

model.num_train = 80000; %����ѵ������������
model.MaxIter = 350; % �����ݶȷ��ĵ�������

model.testnum = 300 ; %ÿ��ѵ��ǰ�� ��������������
model.traintestnum = 1500 ; %ÿ����ѵ��ǰ�� ��������������
model.tick = 15 ; %ѵ��ʱ�Ŀ̶�
model.itn =  30 ; %ÿ��ѵ������������
model.step = 60 ; %ѵ����������
model.interval = model.step/7;  %ÿ��������һ�δ���
model.reservation = 0.000004;    %ÿ�����ٱ�������������
model.rate = 0.30;    %ÿ�������������ű���
model.itreservation = 0.2;    %ÿ�����ٱ����ĵ�����������

model.logn = 0 ; %��ʼ����־����
end

