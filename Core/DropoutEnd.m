function [ model ] = DropoutEnd( model )
%DROPOUTSTART Summary of this function goes here
%   Detailed explanation goes here


num_sublayer = length(model.sublayer); %获得有几层.


for i = num_sublayer:-1: 2
   num_subnet = length(model.sublayer{i}.subnet); %获取当前层subnet的个数
   for j = 1:num_subnet 
        model.sublayer{i}.subnet{j}.model=DropoutNetEnd( model.sublayer{i}.subnet{j}.model );
   end
end
model.OnTrain = 0;
end

function [ model ]=DropoutNetEnd(model)
 num = length(model.Layer);
 for i = num-1:-1: 1
      t = model.Layer{i};
    cur = model.Layer{i}.type;
    
    if strcmp(cur,'ANN')
        tmp = model.Layer{i}.w.*model.Layer{i}.mask;
        tmp = tmp.^2;
        mm = max(tmp(:));
        if mm > 2
        end
        
        model.Layer{i}.mask = ones(size(model.Layer{i}.w));      
    end
 end



end