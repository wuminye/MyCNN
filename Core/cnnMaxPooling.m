function [ featureMap , MaxIndex ] = cnnMaxPooling(inputFeature, kernel ,w,b  )
%CNNPOOLING Summary of this function goes here
%  目前  Pooling 输出没有激活函数

assert(kernel.x <= size(inputFeature ,1), ['Kernel is too large...', '']);
assert(kernel.y <= size(inputFeature ,2), ['Kernel is too large...', '']);

kdx = kernel.x;
kdy = kernel.y;
num = size(inputFeature ,3);
featureMap = zeros(floor(size(inputFeature ,1)/kdx),...
              floor(size(inputFeature ,2)/kdy),size(inputFeature ,3));
MaxIndex = zeros(size(inputFeature));
for i = 1 : num
   
   for x = 1:size(inputFeature ,1)/kdx 
       for y = 1:size(inputFeature ,2)/kdy 
           pacth = inputFeature((x-1)*kdx+1:(x-1)*kdx+kdx,...
                   (y-1)*kdy+1:(y-1)*kdy+kdy, i);
           [featureMap(x,y,i),ind] = max(pacth(:));
           tmp = zeros(size(pacth(:)));
           tmp(ind) = 1;
           tmp =reshape(tmp,kdx,kdy);
           MaxIndex((x-1)*kdx+1:(x-1)*kdx+kdx,...
                   (y-1)*kdy+1:(y-1)*kdy+kdy, i) = tmp;
       end
   end
    
    %{
    tmp = conv2(inputFeature(:,:,i), ones(kdx,kdy),'valid'); 
    featureMap(:,:,i) =w(i).*tmp(1:kdx:end,1:kdy:end)/(kdx*kdy)+b(i);
    %}
end



end

