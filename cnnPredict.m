function [ ress ,X ] = cnnPredict( model,data)
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
X = zeros(36,32,1,0);
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
     ress{j} = b;
     
     %figure;
     [tX] = splitIMG(model,res{1},b);
     temp = X;
     X = zeros(size(tX,1),size(tX,2),size(tX,3),size(tX,4)+size(temp,4));
     X(:,:,1,1:size(temp,4)) = temp;
     X(:,:,1,size(temp,4)+1:end) = tX;
     
     %{
     b(b<=0.8) = 0;
     disp(sum(sum(b>0))); 
     figure;
     imshow(b);
     %}
     scale = scale*0.7;
end

    
end


function [X]=splitIMG(model,img,data)

[sx ,sy] = size(img);
[dx , dy] = size(data);
[xx ,yy] = meshgrid([1:dx],[1:dy]);
xx = xx';
yy = yy';

rx = 36;
ry = 32;

X = zeros(rx,ry,1,0);

th = 0.1;

x = xx(data>=th);
y = yy(data>=th);

N = size(x,1);



for i = 1:floor(N)
    tx = x(i);
    ty = y(i);
    
    cx = ceil(tx*sx/dx - rx/2);
    cy = ceil(ty*sy/dy - ry/2);
    
    if cx<1 || cy<1 || cx+rx-1>sx || cy+ry-1>sy
        continue;
    end
    
    
    res = cnnCalcnet( model, img(cx:cx+rx-1,cy:cy+ry-1));
    rr = res{end};
    
    if rr(1)>0.2
       fprintf('P / N-- %.6f\t/\t%.6f\n',data(tx,ty),rr(1));
       X(:,:,1,end+1) = img(cx:cx+rx-1,cy:cy+ry-1);
       %{
       close all;
       imshow( X(:,:,1,end));
       figure;
       ShowLayer( model, X(:,:,1,end) ,[0 1]);
       pause;
       %}
    end
    
    
    
end



end

