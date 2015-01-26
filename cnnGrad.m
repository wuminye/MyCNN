function [ res ] = cnnGrad( model, data , y ,m )
%CNNGRAD Summary of this function goes here
%   Detailed explanation goes here
num = length(model.Layer);
res = cell(num,1);

%最后一层的误差值,要保证最后两层都是一维的
te1 = data{num}(1,1,:);
te2 = data{num-1}(1,1,:);
res{num}.t = (te1(:) - y)./m;
res{num}.b = res{num}.t;
res{num}.w = (te1(:) - y)*te2(:)'./m;

for i = num-1:-1: 2
    t = model.Layer{i};
    cur = t.type;
    nex =  model.Layer{i+1}.type;
    res{i}.b = 0;
    res{i}.t = 0;
    res{i}.w = 0;
      
    if strcmp(cur,'Pooling') && strcmp(nex,'Conv')   
       res{i}.t = zeros(model.Layer{i}.out);
       
       for p = 1:size(res{i}.t,3)
           for q = 1:size(res{i+1}.t,3)
               if model.Layer{i+1}.connector(q,p)~=1
                   continue;
               end
               res{i}.t(:,:,p) = res{i}.t(:,:,p) + ...
                        conv2(res{i+1}.t(:,:,q),model.Layer{i+1}.w(:,:,p,q),'full');
           end
       end 
       
    end
    
    if strcmp(cur,'Pooling') && strcmp(nex,'ANN')
        res{i}.t = model.Layer{i+1}.w'*res{i+1}.t;  %.*(data{i}.*(1-data{i}));Pooling层没有激活函数
        res{i}.b = reshape(res{i}.t,1,1,[]);% 转化成featureMap
        res{i}.t = res{i}.b;
        %res{i}.w = res{i}.b*data{i-1}'; Pooling层没有权值
    end
    
    if (strcmp(cur,'Pooling') || strcmp(cur,'Conv')) && strcmp(nex,'Pooling')
        k = model.Layer{i+1}.kernel;
        B = ones(k.x,k.y);
        %初始化误差featuremap
        res{i}.t = zeros(model.Layer{i}.out);
        %计算有效误差矩阵的大小
        x = size(res{i+1}.b,1)*k.x;
        y = size(res{i+1}.b,2)*k.y;
        
        for j = 1 : size(res{i+1}.b,3)
            %有效误差矩阵
            res{i}.t(1:x,1:y,j) = kron(res{i+1}.b(:,:,j) , B)./(k.x*k.y);
            %res{i}.t(1:x,1:y,j) = kron(res{i+1}.b(:,:,j) , B);
        end
        
        %计算卷积层的核函数梯度
        if strcmp(cur,'Conv')
            
            %修正卷积层的误差，要乘以导数。
            res{i}.t = res{i}.t.*(data{i}.*(1-data{i}));
            
            %Initialize Conv Kernel
            res{i}.w = zeros(size(model.Layer{i}.w));
            res{i}.b = zeros(size(model.Layer{i}.b));
            for q = 1 : size(res{i}.w,4)
               for p = 1 : size(res{i}.w,3)
                   res{i}.w(:,:,p,q) = conv2(data{i-1}(:,:,p), rot90(rot90(res{i}.t(:,:,q))),'valid');
                   %------------这里要不要再翻转回来呢?????????????????????
                   %res{i}.w(:,:,p,q) = rot90(rot90(conv2(data{i-1}(:,:,p), rot90(rot90(res{i}.t(:,:,q))),'valid')));
               end
               tem =res{i}.t(:,:,q);
               res{i}.b(q) = sum(tem(:));
            end
            
        else
             res{i}.b =  res{i}.t; % 非卷积层
        end
    end
    
    if strcmp(cur,'ANN')
        res{i}.t = model.Layer{i+1}.w'*res{i+1}.t.*(data{i}.*(1-data{i}));
        res{i}.b = res{i}.t;
        res{i}.w = res{i}.b*data{i-1}';
    end
end

end

