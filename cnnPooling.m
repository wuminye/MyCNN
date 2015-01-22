function [ featureMap ] = cnnPooling(inputFeature, kernel )
%CNNPOOLING Summary of this function goes here
%   Detailed explanation goes here

assert(kernel.x <= size(inputFeature ,1), ['Kernel is too large...', '']);
assert(kernel.y <= size(inputFeature ,2), ['Kernel is too large...', '']);

kdx = kernel.x;
kdy = kernel.y;
num = size(inputFeature ,3);
featureMap = zeros(floor(size(inputFeature ,1)/kdx),...
              floor(size(inputFeature ,2)/kdy),size(inputFeature ,3));

for i = 1 : num
   for x = 1:size(inputFeature ,1)/kdx 
       for y = 1:size(inputFeature ,2)/kdy 
           pacth = inputFeature((x-1)*kdx+1:(x-1)*kdx+kdx,...
                   (y-1)*kdy+1:(y-1)*kdy+kdy, i);
           featureMap(x,y,i) = mean(pacth(:));
       end
   end
end



end

