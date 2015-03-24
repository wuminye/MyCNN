   function [ featureMap ] = cnnConvs(  inputFeature , w , b , connector , stride )

assert(size(w ,4) == size(b ,1), ['Dims of w and b error  ', '']);
assert(size(connector ,2) == size(inputFeature ,3), ['Dims of inputFeature or connector error  ', '']);
assert(size(connector ,1) == size(w ,4), ['Dims of featureMap or connector error  ', '']);

old_num = size(inputFeature ,3);
num = size(w ,4);
x = size(inputFeature ,1)/stride;
y = size(inputFeature ,2)/stride;

featureMap = zeros(x ,y ,num);


for cf = 1 : num
    for lf = 1 : old_num
        if connector(cf,lf)==0
            continue;
        end
        tem = conv2(inputFeature(:,:,lf),rot90(w(:,:,lf,cf),2),'same');
        tem = tem(1:stride:end,1:stride:end);
        featureMap(:,:,cf) = featureMap(:,:,cf) +tem;
    end
    featureMap(:,:,cf) = ActiveFunction(featureMap(:,:,cf) + b(cf));
end

end

