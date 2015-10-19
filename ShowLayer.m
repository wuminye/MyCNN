function [ res ] = ShowLayer( model, data ,y)
%可视化网络的中间二维图像计算结果
%data为featureMap的格式
%close all;

sx = 3;
sy = 4;

tic;
res = cnnCalcSubnet(model,data,0);
toc;
num = length(model.Layer);
cnt=1;
%colormap(gray);
axis image off
for i = 1 : num
    axis image off
    cur = model.Layer{i}.type;
    if strcmp(cur,'Conv') || strcmp(cur,'Pooling') || strcmp(cur,'ANN')  || strcmp(cur,'Convs')|| strcmp(cur,'MaxPooling')
       tem = reshape(res{i},[],size(res{i},3));
       [h,im]=displayData(tem',size(res{i},2)); 
       subplot(sx,sy,cnt);
       imagesc(im ,[min(im(:)) max(im(:))]);
       %imshow(im);
       cnt =cnt+1;
    end

end

tem = reshape(res{num},[],size(res{num},3));
%[h,im]=displayData(tem,size(res{num},3)); 
subplot(sx,sy,cnt);
%imagesc(im ,[0 1]);
bar(tem,0.4,'histc');
cnt =cnt+1;

subplot(sx,sy,cnt);
 imagesc(data(:,:),[0 1]);
 
cnt =cnt+1;

subplot(sx,sy,cnt);
axis image off
imagesc(reshape(y,[],size(y,2)),[0 1]);
axis image off
%[a,b]=max(res{num});
%disp((res{num}(:))');
%fprintf('ANS: %i\n',b);
end

