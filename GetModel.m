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

   num_layer = 7;
   Layer = cell(num_layer,1);

   Layer{1}.type = 'Input';
   Layer{1}.out = input;

     
   Layer{2}.type = 'Convs';
   Layer{2}.kernelsize = [5 5];
   Layer{2}.mapnum  =  4;
   Layer{2}.stride = 2 ;    %  18 16


   Layer{3}.type = 'Convs';
   Layer{3}.kernelsize = [5 5];
   Layer{3}.mapnum  =  8;
   Layer{3}.stride = 2 ;    %  9 8

   Layer{4}.type = 'Conv';
   Layer{4}.kernelsize = [5 5];
   Layer{4}.mapnum  =   16;  %5 4
   
   Layer{5}.type = 'Conv';
   Layer{5}.kernelsize = [5 4];
   Layer{5}.mapnum  =   32;  %1 1

   Layer{6}.type = 'ANN';
   Layer{6}.out = [28 1];

   Layer{7}.type = 'SoftMax';
   Layer{7}.out = [2 1];

end

model = InitModel(Layer);

model.lambda = 0.003;
model.dataname = name;  %���ݿ�����

model.num_train = 60000; %����ѵ������������
model.MaxIter = 350; % �����ݶȷ��ĵ�������

model.testnum = 400 ; %ÿ��ѵ��ǰ�� ��������������
model.traintestnum = 1500 ; %ÿ����ѵ��ǰ�� ��������������
model.tick = 16 ; %ѵ��ʱ�Ŀ̶�
model.itn =  27 ; %ÿ��ѵ��������������
model.step = 45 ; %ѵ����������
model.interval = model.step/7;  %ÿ��������һ�δ�����
model.reservation = 0.0005;    %ÿ�����ٱ�������������
model.rate = 0.1;    %ÿ�������������ű���
model.itreservation = 0.2;    %ÿ�����ٱ����ĵ�����������

model.logn = 0 ; %��ʼ����־����
end
