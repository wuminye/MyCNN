function [ ress ] = cnnPredict( model,data)
%CNNPREDICT Summary of this function goes here
%   Detailed explanation goes here
if size(data,1)==1   %Œƒº˛ ‰»Î
    data = imread(data);
    data = rgb2gray(data);
    data = imresize(data,1500/max(size(data)));
    data = double(data)/255;
end   

scale = 1;
step = 4;
ress =cell(step,1);

for j = 1:step
     res = cell(length(model.Layer),1);
     res{1} = imresize(data,scale);
     endmark = 1;
     for i  = 2 : length(model.Layer)
         cur = model.Layer{i}.type;
         if strcmp(cur,'Reshape')
             res{i} = cnnReshape(res{i-1},model.Layer{i}.kernelsize); 
         end
         if strcmp(cur,'SoftMax')
            res{i} = cnnSoftMax(res{i-1},model.Layer{i}.w);
         end
         if strcmp(cur,'Conv')
             res{i} = cnnConv(res{i-1},model.Layer{i}.w,model.Layer{i}.b,...
                              model.Layer{i}.connector);
         end
          if strcmp(cur,'Pooling')
             res{i} = cnnPooling(res{i-1}, model.Layer{i}.kernel );
          end
         if strcmp(cur,'ANN')
             endmark = i-1;
             break;
         end
     end
     temp = zeros(size(res{endmark},3),size(res{endmark},1)*size(res{endmark},2));
     for p = 1 : size(res{endmark},3)
        temp(p,:) = reshape(res{endmark}(:,:,p),1,[]); 
     end
     i = endmark+1;
     res{i} =ActiveFunction(model.Layer{i}.w*temp + repmat(model.Layer{i}.b,1,size(temp,2)));
     i = i + 1;
     res{i} = exp( model.Layer{i}.w*res{i-1});
     res{i} = res{i}./repmat(sum(res{i}),2,1);
     b = reshape(res{i}(1,:),size(res{endmark},1),[]);
     b(b<=0.7) = 0;
     disp(sum(sum(b>0))); 
     figure;
     imshow(b);
     scale = scale*0.7;
end

    
end

