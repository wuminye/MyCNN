function [ ress ,X ,scales] = cnnPredict( model,data , state)
%CNNPREDICT Summary of this function goes here
%   Detailed explanation goes here

if ~exist('state', 'var')
    state = 0;
end

if size(data,1)==1   %文件输入
    data = imread(data);
    data = rgb2gray(data);
    data = imresize(data,800/max(size(data)));
    data = double(data)/255;
end   

scale = 1;
step = 7;
ress =cell(step,1);
X = zeros(36,32,1,0);
scales = zeros(0,1);
for j = 1:step
    
    %过载保护
    if size(X,4)>800 
        break;
    end
    
    fprintf('%d\n',j);
     res = cell(length(model.Layer),1);
     res{1} = imresize(data,scale);
     scales(end+1,1) = scale;
     %{
     imshow( res{1});
     hold on;
      rectangle('Position', ...
            [1, 1, 32, 36], ...
            'Curvature', 0.4, 'LineWidth',1, 'EdgeColor', 'blue');
     figure;
     %}
     endmark = 1;
     for i  = 2 : length(model.Layer)
       
         cur = model.Layer{i}.type;
         if strcmp(cur,'Reshape')
            % res{i} = cnnReshape(res{i-1},model.Layer{i}.kernelsize); 
             window = model.Layer{i-1}.out;
             res{i} = zeros(size(res{i-1},1)-window(1)+1,size(res{i-1},2)-window(2)+1,model.Layer{i}.kernelsize(3));
             for x = 1:size(res{i-1},1)-window(1)+1
                 for y = 1 : size(res{i-1},2)-window(2)+1
                     tmp = res{i-1}(x:x+window(1)-1,y:y+window(2)-1,:);
                    
                     res{i}(x,y,:) = reshape(tmp,model.Layer{i}.kernelsize);
                 end
             end
         end
         if strcmp(cur,'SoftMax')
            res{i} = cnnSoftMax(res{i-1},model.Layer{i}.w);
         end
         if strcmp(cur,'Conv')
             res{i} = cnnConv(res{i-1},model.Layer{i}.w,model.Layer{i}.b,...
                              model.Layer{i}.connector);
         end
         if strcmp(cur,'Convs')
             res{i} = cnnConvs(res{i-1},model.Layer{i}.w,model.Layer{i}.b,...
                              model.Layer{i}.connector,model.Layer{i}.stride);
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
     
     
     %figure;
     rate = [0.2 0.3]'; % 默认检测误报样本
      
     ress{j} = b;
      
     if state == 0
         [tX] = splitIMG(model,res{1},b,rate);
         temp = X;
         X = zeros(size(tX,1),size(tX,2),size(tX,3),size(tX,4)+size(temp,4));
         X(:,:,1,1:size(temp,4)) = temp;
         X(:,:,1,size(temp,4)+1:end) = tX;
     end
     %{
     b(b<=0.8) = 0;
     disp(sum(sum(b>0))); 
     figure;
     imshow(b);
     %}
 
     scale = scale*0.72;
end

    
end


function [X]=splitIMG(model,img,data,rate)

[sx ,sy] = size(img);
[dx , dy] = size(data);
[xx ,yy] = meshgrid([1:dx],[1:dy]);
xx = xx';
yy = yy';

rx = 36;
ry = 32;

X = zeros(rx,ry,1,0);

th = rate(1);

x = xx(data>=th);
y = yy(data>=th);

N = size(x,1);



for i = 1:floor(N)
    tx = x(i);
    ty = y(i);
    
     %坐标转换到原图
     cx = ceil(tx*(sx-36)/dx - rx/2 + 16);
     cy = ceil(ty*(sy-32)/dy - ry/2 + 16);
    
    if cx<1 || cy<1 || cx+rx-1>sx || cy+ry-1>sy
        continue;
    end
    
    
    res = cnnCalcnet( model, img(cx:cx+rx-1,cy:cy+ry-1));
    rr = res{end};
   % imshow(img(cx:cx+rx-1,cy:cy+ry-1));
    if rr(1)>rate(2)
       if rate(2) <0.5
          fprintf('P / N-- %.6f\t/\t%.6f\n',data(tx,ty),rr(1));
       end
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

