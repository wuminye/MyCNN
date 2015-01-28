function [ ] = ShowLayer( model, data )
%���ӻ�������м��άͼ�������
%dataΪfeatureMap�ĸ�ʽ
close all;
res = cnnCalcnet(model,data);
num = length(model.Layer);

for i = 1 : num
    cur = model.Layer{i}.type;
    if strcmp(cur,'Conv') || strcmp(cur,'Pooling')
       tem = reshape(res{i},[],size(res{i},3));
       figure;
       displayData(tem');        
    end

end
[a,b]=max(res{num});
disp((res{num}(:))');
fprintf('ANS: %i\n',b);
end

