function [ res ] = cnnCalcNetReg( model )
%��������������

res = 0;
num_sublayer = length(model.sublayer); %����м���.

for i = 2 : num_sublayer
    num_subnet = length(model.sublayer{i}.subnet); %��ȡ��ǰ��subnet�ĸ���
    for j = 1 :  num_subnet
        res = res + cnnCalcReg(model.sublayer{i}.subnet{j}.model);
    end
end

end

