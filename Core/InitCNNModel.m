function [ model ] = InitCNNModel()
% �涨ÿ��subnet�ĵ�һ���� input��

%{
   input = [36 32 1];

   
%---------------------------------------------
% subnet 1-1
%---------------------------------------------
   Layer = cell(0,1);

   Layer{end+1}.type = 'Input';
   Layer{end}.out = input;
     
   Layer{end+1}.type = 'Conv';
   Layer{end}.kernelsize = [5 5];
   Layer{end}.mapnum  =   6;  %32 28
   
   Layer{end+1}.type = 'Pooling';
   Layer{end}.kernelsize = [2 2];  %  16 14 
   
   model_1_1 = InitSubnet( Layer );
   
%---------------------------------------------
% subnet 2-1
%---------------------------------------------
   Layer = cell(0,1);
   
   Layer{end+1}.type = 'Input';
   Layer{end}.out = [16 14 6];
   
   Layer{end+1}.type = 'Convs';
   Layer{end}.kernelsize = [5 5];
   Layer{end}.mapnum  =  9;
   Layer{end}.stride = 2 ;    %  8 7
   
   model_2_1 = InitSubnet( Layer );
   
%---------------------------------------------
% subnet 2-2
%---------------------------------------------
   Layer = cell(0,1);
   
   Layer{end+1}.type = 'Input';
   Layer{end}.out = [16 14 6];
   
   Layer{end+1}.type = 'Pooling';
   Layer{end}.kernelsize = [2 2];  %  8 7 

   
   model_2_2 = InitSubnet( Layer );   
   
%---------------------------------------------
% subnet 3-1
%---------------------------------------------
   Layer = cell(0,1);   
   
   Layer{end+1}.type = 'Input';
   Layer{end}.out = [8 7 15];
   
   Layer{end+1}.type = 'Conv';
   Layer{end}.kernelsize = [3 3];
   Layer{end}.mapnum  =   26;  %6 5
   
   Layer{end+1}.type = 'Conv';
   Layer{end}.kernelsize = [6 5];
   Layer{end}.mapnum  =   26;  % 1 1

   Layer{end+1}.type = 'ANN';
   Layer{end}.out = [52 1];
   
   Layer{end+1}.type = 'SoftMax';
   Layer{end}.out = [2 1];
 
   model_3_1 = InitSubnet( Layer ); 
   
   model_3_1.layer{3}.connector = eye(26,26);  
   
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

%}

input = [28 28 1];

%---------------------------------------------
% subnet 1-1
%---------------------------------------------
   Layer = cell(0,1);

   
   Layer{end+1}.type = 'Input';
   Layer{end}.out = input;
        
   Layer{end+1}.type = 'Conv';
   Layer{end}.kernelsize = [5 5];
   Layer{end}.mapnum  =   7;  %24 24s
   
   Layer{end+1}.type = 'Pooling';
   Layer{end}.kernelsize = [2 2];  %  12 12 
   
   Layer{end+1}.type = 'Conv';
   Layer{end}.kernelsize = [5  5 ];
   Layer{end}.mapnum  =   20;  %8 8
   
   Layer{end+1}.type = 'Pooling';
   Layer{end}.kernelsize = [2 2];  %  4 4
   
 
   
     Layer{end+1}.type = 'Reshape';
   Layer{end}.kernelsize = [1 1 4*4*20];

      
   
   
   Layer{end+1}.type = 'ANN';
   Layer{end}.out = [300 1];

     
   
   Layer{end+1}.type = 'SoftMax';
   Layer{end}.out = [10 1];

   %{
   Layer{end+1}.type = 'Input';
   Layer{end}.out = input;
   
   Layer{end+1}.type = 'Pooling';
   Layer{end}.kernelsize = [2 2];  %  14 14
   
    Layer{end+1}.type = 'Pooling';
   Layer{end}.kernelsize = [2 2];  %  14 14
   
   Layer{end+1}.type = 'Reshape';
   Layer{end}.kernelsize = [1 1 7*7];
   Layer{end}.mapnum  =   2;  % 1 1
   
   Layer{end+1}.type = 'SoftMax';
   Layer{end}.out = [10 1];
   %}
   model_1_1 = InitSubnet( Layer );
   

%====================================================

num_sublayer = 1;

model.sublayer = cell(num_sublayer+1,1);
% model.sublayer{1}.subnet{1} Ҫ���� ���������
model.sublayer{1}.subnet{1} = 0;
model.sublayer{2}.subnet{1}.model = model_1_1;
%-------- CONNECTION --------------------
model.sublayer{2}.connect = ones(1,length(model.sublayer{2}.subnet));
model.type = 'big';


model.lambda = 0.003; 
%model.dataname = name;  %���ݿ����� 

model.num_train = 60000; %����ѵ������������ 
model.MaxIter = 20; % �����ݶȷ��ĵ������� 

 
model.testnum = 500 ; %ÿ��ѵ��ǰ�� �������������� 
model.traintestnum = 1000 ; %ÿ����ѵ��ǰ�� �������������� 
model.tick = 15 ; %ѵ��ʱ�Ŀ̶� 
model.itn =  50 ; %ÿ��ѵ������������ 
model.step = 36 ; %ѵ���������� 
model.interval = model.step/7;  %ÿ��������һ�δ��� 
model.reservation = 0.0014;    %ÿ�����ٱ������������� 
model.rate = 0.03;    %ÿ�������������ű��� 
model.itreservation = 0.35;    %ÿ�����ٱ����ĵ����������� 

 
model.logn = 0 ; %��ʼ����־���� 



end

