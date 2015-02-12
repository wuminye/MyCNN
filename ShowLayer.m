function [ ] = ShowLayer( model, data ,y)
%可视化网络的中间二维图像计算结果
%data为featureMap的格式
close all;
tic;
res = cnnCalcnet(model,data);
toc;
num = length(model.Layer);
cnt=1;
colormap(gray);
for i = 1 : num
    cur = model.Layer{i}.type;
    if strcmp(cur,'Conv') || strcmp(cur,'Pooling') 
       tem = reshape(res{i},[],size(res{i},3));
       [h,im]=displayData(tem',size(res{i},2)); 
       subplot(3,4,cnt);
       imagesc(im ,[0 1]);
       %imshow(im);
       cnt =cnt+1;
    end

end

tem = reshape(res{num},[],size(res{num},3));
[h,im]=displayData(tem,17); 
subplot(3,4,cnt);
imagesc(im ,[0 1]);
cnt =cnt+1;

subplot(3,4,cnt);
 imagesc(data(:,:),[0 1]);
 
cnt =cnt+1;

subplot(3,4,cnt);
imagesc(reshape(y,[],size(y,2)),[0 1]);
axis image off
%[a,b]=max(res{num});
%disp((res{num}(:))');
%fprintf('ANS: %i\n',b);
end

