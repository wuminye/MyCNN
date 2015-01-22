function [ featureMap ] = cnnConv( inputFeature , w , b , connector)
%CNNCONV Summary of this function goes here
%   Detailed explanation goes here
%   ===========================================
%   inputFeature : lastLayerFeature  (ox , oy , old_num)
%   w :   weights of kernel         (a, b, num)
%   b :   bias                      (x, y, num)
%   connector: the way of  featuremaps connections between current layer
%              and last layer  (i , j)==1 current i-th featuremap connect
%              j-th featuremap of last layer 
%  -------------------------------------------------
%  
assert(size(w ,3) == size(b ,3), ['Dims of w and b error  ', '']);
assert(size(connector ,2) == size(inputFeature ,3), ['Dims of inputFeature or connector error  ', '']);
assert(size(connector ,1) == size(w ,3), ['Dims of featureMap or connector error  ', '']);
assert(size(b ,1) == size(inputFeature ,1)-size(w ,1)+1, ['valid dims of featuremap ', '']);
assert(size(b ,2) == size(inputFeature ,2)-size(w ,2)+1, ['valid dims of featuremap ', '']);
%  -------------------------------------------------
old_num = size(inputFeature ,3);
ox = size(inputFeature ,1);
oy = size(inputFeature ,2);

num = size(w ,3);
x =  size(b ,1);
y = size(b,2);
% do Conv, [valid] dim of each featuremap to inputFeature
featureMap = zeros(x ,y ,num);

for cf = 1 : num
    for lf = 1 : old_num
        if connector(cf,lf)==0
            continue;
        end
        tem = conv2(inputFeature(:,:,lf),rot90(w(:,:,cf),2),'valid');
        featureMap(:,:,cf) = featureMap(:,:,cf) +tem;
    end
    featureMap(:,:,cf) = ActiveFunction(featureMap(:,:,cf) + b(:,:,cf));
end

end

