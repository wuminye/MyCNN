function [ model ] = InitCNNModel()
% �涨ÿ��subnet�ĵ�һ���� input��


   input = [36 32 1];

   
%---------------------------------------------
% subnet 1-1
%---------------------------------------------
   Layer = cell(0,1);

   Layer{end+1}.type = 'Input';
   Layer{end}.out = input;
     
   Layer{end+1}.type = 'Convs';
   Layer{end}.kernelsize = [7 7];
   Layer{end}.mapnum  =  3;
   Layer{end}.stride = 2 ;    %  18 16

   Layer{end+1}.type = 'Conv';
   Layer{end}.kernelsize = [5 5];
   Layer{end}.mapnum  =   8;  %14 12
   
   model_1_1 = InitSubnet( Layer );
   
%---------------------------------------------
% subnet 2-1
%---------------------------------------------
   Layer = cell(0,1);
   
   Layer{end+1}.type = 'Input';
   Layer{end}.out = [14 12 8];
   
   Layer{end+1}.type = 'Convs';
   Layer{end}.kernelsize = [4 4];
   Layer{end}.mapnum  =  12;
   Layer{end}.stride = 2 ;    %  7 6 
   
   model_2_1 = InitSubnet( Layer );
   
%---------------------------------------------
% subnet 2-2
%---------------------------------------------
   Layer = cell(0,1);
   
   Layer{end+1}.type = 'Input';
   Layer{end}.out = [14 12 8];
   
   Layer{end+1}.type = 'Pooling';
   Layer{end}.kernelsize = [2 2];  %  7 6 

   
   model_2_2 = InitSubnet( Layer );   
   
%---------------------------------------------
% subnet 3-1
%---------------------------------------------
   Layer = cell(0,1);   
   
   Layer{end+1}.type = 'Input';
   Layer{end}.out = [7 6 20];
   
   Layer{end+1}.type = 'Conv';
   Layer{end}.kernelsize = [4 4];
   Layer{end}.mapnum  =   25;  %4 3
   
   Layer{end+1}.type = 'Conv';
   Layer{end}.kernelsize = [4 3];
   Layer{end}.mapnum  =   35;  % 1 1

   Layer{end+1}.type = 'ANN';
   Layer{end}.out = [30 1];

   Layer{end+1}.type = 'SoftMax';
   Layer{end}.out = [2 1];
 
   model_3_1 = InitSubnet( Layer );   
   
%====================================================

num_sublayer = 3;

model.sublayer = cell(num_sublayer+1,1);

% model.sublayer{1}.subnet{1} Ҫ���� ���������
model.sublayer{1}.subnet{1} = 0;
model.sublayer{2}.subnet{1}.model = model_1_1;
model.sublayer{3}.subnet{1}.model = model_2_1;
model.sublayer{3}.subnet{2}.model = model_2_2;
model.sublayer{4}.subnet{1}.model = model_3_1;
%-------- CONNECTION --------------------
model.sublayer{2}.connect = ones(1,length(model.sublayer{2}.subnet));
model.sublayer{3}.connect = ones(length(model.sublayer{2}.subnet),...
                             length(model.sublayer{3}.subnet));
model.sublayer{4}.connect = ones(length(model.sublayer{3}.subnet),...
                              length(model.sublayer{4}.subnet));
%============================================================

model.lambda = 0.002; 
%model.dataname = name;  %���ݿ����� 

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

