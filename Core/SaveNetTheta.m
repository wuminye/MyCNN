function [ theta ] = SaveNetTheta( model )
%��ȡģ�Ͳ���������

num_sublayer = length(model.sublayer); %����м���.
theta = [];

for i = 2 : num_sublayer
    num_subnet = length(model.sublayer{i}.subnet); %��ȡ��ǰ��subnet�ĸ���
    for j = 1 :  num_subnet
        theta = [theta ; SaveTheta(model.sublayer{i}.subnet{j}.model)];       
    end
end

end

