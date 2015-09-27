function [ model ] = DropoutStart( model )
%DROPOUTSTART Summary of this function goes here
%   Detailed explanation goes here

if  model.Dropout == 0
    fprintf('Disable Dropout\n');
    return ;
end
model.OnTrain = 1;
%fprintf('------Enable Dropout-------\n');

num_sublayer = length(model.sublayer); %获得有几层.


for i = num_sublayer:-1: 2
   num_subnet = length(model.sublayer{i}.subnet); %获取当前层subnet的个数
   for j = 1:num_subnet 
       model.sublayer{i}.subnet{j}.model=DropoutNetStart( model.sublayer{i}.subnet{j}.model );
   end
end

end

function [ model ]=DropoutNetStart(model)
 num = length(model.Layer);
 for i = num-1:-1: 1
      t = model.Layer{i};
    cur = model.Layer{i}.type;
    
    if strcmp(cur,'ANN')
        if t.dropout.enable ~= 1
            continue;
        end
    
        n = size(model.Layer{i}.w(:),1);
        ind = rand(n,1);
        ind(ind<=model.Layer{i}.dropout.p) = 0;
        ind(ind>model.Layer{i}.dropout.p) =1;
        ind = ~logical(ind);
        model.Layer{i}.mask(ind) = 0;
        
    end
 end



end