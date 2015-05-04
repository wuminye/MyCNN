function [ model ] = ModelInit( n )

   input = [1 1 n];
   Layer = cell(0,1);

   Layer{end+1}.type = 'Input';
   Layer{end}.out = input;
        
   Layer{end+1}.type = 'ANN';
   Layer{end}.out = [18 1];
   
   Layer{end+1}.type = 'SoftMax';
   Layer{end}.out = [2 1];
   
   model_1_1 = InitSubnet( Layer );
   
   num_sublayer = 1;

model.sublayer = cell(num_sublayer+1,1);
% model.sublayer{1}.subnet{1} Ҫ���� ���������
model.sublayer{1}.subnet{1} = 0;
model.sublayer{2}.subnet{1}.model = model_1_1;
%-------- CONNECTION --------------------
model.sublayer{2}.connect = ones(1,length(model.sublayer{2}.subnet));
model.type = 'big';


model.lambda = 0.03; 
%model.dataname = name;  %���ݿ����� 

model.num_train = 200000; %����ѵ������������ 
model.MaxIter = 20; % �����ݶȷ��ĵ������� 

 
model.testnum = 200 ; %ÿ��ѵ��ǰ�� �������������� 
model.traintestnum = 4000 ; %ÿ����ѵ��ǰ�� �������������� 
model.tick = 15 ; %ѵ��ʱ�Ŀ̶� 
model.itn =  25 ; %ÿ��ѵ������������ 
model.step = 30 ; %ѵ���������� 
model.interval = model.step/7;  %ÿ��������һ�δ��� 
model.reservation = 0.00015;    %ÿ�����ٱ������������� 
model.rate = 0.009;    %ÿ�������������ű��� 
model.itreservation = 0.3;    %ÿ�����ٱ����ĵ����������� 

 
model.logn = 0 ; %��ʼ����־���� 

end

