function [ images ] = NormalizeIMG( images)
%����featuremap��ʽ��ͼƬ


N = size(images,4);

parfor i = 1:N
   t = images(:,:,:,i);
   t = mean(t(:));
 
   images (:,:,:,i) = images(:,:,:,i) - t;
end

end

