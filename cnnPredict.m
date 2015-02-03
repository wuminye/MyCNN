function [ res ] = cnnPredict( model,data)
%CNNPREDICT Summary of this function goes here
%   Detailed explanation goes here
if size(data,1)==1   %Œƒº˛ ‰»Î
    data = imread(data);
    data = rgb2gray(data);
    data = double(data)/255;
end   

step = 4;

scale = [7.6];

for i = 1 : size(scale,2)
    img = imresize(data,1/scale(i));
    [x , y] = size(img);
    
    for dx = 1 : step : x-20
        for dy = 1 : step : y-20
            patch = img(dx:dx + 20 -1,dy:dy + 20 - 1);
            
            res = cnnCalcnet(model,patch);
            output = res{length(res)}(:);
            [q,ar] = max(output);
            if ar == 1
                close all;
                imshow(patch);
                figure;
                imshow(data);
                rectangle('Position',[dy*scale(i),dx*scale(i),20*scale(i),20*scale(i)]) ;
               pause;
                
            end
           
        end
        
        
    end
   
    
    
end

end

