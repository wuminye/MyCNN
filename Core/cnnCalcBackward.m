function [ res ] = cnnCalcBackward( model, data , errdata )

num_sublayer = length(model.sublayer); %����м���.
res = cell(num_sublayer,1);

for i = num_sublayer:-1: 2
 
     num_subnet = length(model.sublayer{i}.subnet); %��ȡ��ǰ��subnet�ĸ���
     
     for j = 1 : num_subnet
         
         
     end
    
    
end    
    
    
end


function [ errdata ] = adhesive(model,data)


end

