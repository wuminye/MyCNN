function [ images ] = NormalizeIMG( images)
%输入featuremap格式的图片


N = size(images,4);

parfor i = 1:N
   t = images(:,:,:,i);
   t = mean(t(:));
 
   images (:,:,:,i) = images(:,:,:,i) - t;
end

end

