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
% model.sublayer{1}.subnet{1} 要声明 ，这层留空
model.sublayer{1}.subnet{1} = 0;
model.sublayer{2}.subnet{1}.model = model_1_1;
%-------- CONNECTION --------------------
model.sublayer{2}.connect = ones(1,length(model.sublayer{2}.subnet));
model.type = 'big';


model.lambda = 0.03; 
%model.dataname = name;  %数据库名称 

model.num_train = 200000; %用于训练的样本数量 
model.MaxIter = 20; % 批量梯度法的迭代次数 

 
model.testnum = 200 ; %每批训练前后 测试样本的数量 
model.traintestnum = 4000 ; %每多批训练前后 测试样本的数量 
model.tick = 15 ; %训练时的刻度 
model.itn =  25 ; %每批训练最大迭代次数 
model.step = 30 ; %训练的批次数 
model.interval = model.step/7;  %每隔多少批一次大检查 
model.reservation = 0.00015;    %每批最少保留的样本比例 
model.rate = 0.009;    %每批样本数量缩放比例 
model.itreservation = 0.3;    %每批最少保留的迭代次数比例 

 
model.logn = 0 ; %初始化日志计数 

end

