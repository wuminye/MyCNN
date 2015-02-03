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
   
model = InitModel(Layer);

model.lambda = 0.002;
model.dataname = name;  %���ݿ�����

model.num_train = 3000; %����ѵ������������
model.MaxIter = 350; % �����ݶȷ��ĵ�������

model.testnum = 200 ; %ÿ��ѵ��ǰ�� ��������������
model.traintestnum = 3000 ; %ÿ����ѵ��ǰ�� ��������������
model.tick = 15 ; %ѵ��ʱ�Ŀ̶�
model.itn =  30 ; %ÿ��ѵ������������
model.step = 50 ; %ѵ����������
model.interval = model.step/7;  %ÿ��������һ�δ���
model.reservation = 0.002;    %ÿ�����ٱ�������������
model.rate = 0.42;    %ÿ�������������ű���
model.itreservation = 0.2;    %ÿ�����ٱ����ĵ�����������

model.logn = 0 ; %��ʼ����־����
end

