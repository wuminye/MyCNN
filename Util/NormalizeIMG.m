function [ imagess ] = NormalizeIMG( images ,scale)
%输入featuremap格式的图片

if ~exist('scale', 'var')
    scale = 1;
end



window = [7 7];
window = floor(window/2);

N = size(images,4);
sx = size(images,1);
sy = size(images,2);
[a b c d] = size(images);
imagess = zeros([a b c d].*[scale scale 1 1]);
parfor i = 1:N
  % tem  = zeros(sx,sy);
   %{
   for x = 1:sx
    for y = 1:sy
        
        patch = images(:,:,1,i);
        patch = patch(max(x-window(1),1):min(x+window(1),sx),...
                      max(y-window(2),1):min(y+window(2),sy) );
        am = mean(patch(:));
        
        tem(x,y) =  (images(x,y,1,i) - am) /std(patch(:),1)/sqrt(length(patch(:)));    
        
    end
   end
   %}
   imagess (:,:,1,i) = imresize(images(:,:,1,i),scale); 
end

end

