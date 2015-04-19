function [ model ] = LoadNetTheta( theta , old_model )
%��ȡ�������е�ģ��

model = old_model;
num_sublayer = length(old_model.sublayer); %����м���.

pos= 1;
for i = 2 : num_sublayer
    num_subnet = length(model.sublayer{i}.subnet); %��ȡ��ǰ��subnet�ĸ���
    for j = 1 :  num_subnet
         n = length(SaveTheta(model.sublayer{i}.subnet{j}.model));  
         
         model.sublayer{i}.subnet{j}.model = ...
             LoadTheta(theta(pos:pos+n-1), model.sublayer{i}.subnet{j}.model);
         
         pos = pos+n;
    end
end


end

