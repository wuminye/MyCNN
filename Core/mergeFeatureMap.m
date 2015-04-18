function [ res ] = mergeFeatureMap( a,b )
%合并featuremap格式数据

res = zeros(size(a,1),size(a,2),size(a,3),size(a,4)+size(b,4));

res(:,:,:,1:size(a,4)) =a;
res(:,:,:,size(a,4)+1:end)=b;

end

