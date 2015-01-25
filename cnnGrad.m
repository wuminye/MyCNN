function [ res ] = cnnGrad( model, data , y ,m )
%CNNGRAD Summary of this function goes here
%   Detailed explanation goes here
num = length(model.Layer);
res = cell(num,1);
res{num}.b = (data{num} - y)./m;
res{num}.w = (data{num} - y)*data{num-1}'./m;
for i = num-1:-1: 2
    t = model.Layer{i};
    cur = t.type;
    nex =  model.Layer{i+1}.type;
    res{i}.b = 0;
    res{i}.w = 0;
       
    
    if strcmp(cur,'Pooling') && strcmp(nex,'ANN')
        res{i}.b = model.Layer{i+1}.w'*res{i+1}.b;  %.*(data{i}.*(1-data{i}));Pooling层没有激活函数
        res{i}.b = reshape(res{i}.b,1,1,size(res{i}.b,1));% 转化成featureMap
        %res{i}.w = res{i}.b*data{i-1}'; Pooling层没有权值
    end
    
    if (strcmp(cur,'Pooling') || strcmp(cur,'Conv')) && strcmp(nex,'Pooling')
        k = model.Layer{i+1}.kernel;
        B = ones(size(k));
        %初始化误差featuremap
        res{i}.b = zeros(size(model.Layer{i-1}.out));
        %计算有效误差矩阵的大小
        x = size(res{i+1}.b,1)*size(k,1);
        y = size(res{i+1}.b,2)*size(k,2);
        
        for j = 1 : size(res{i+1}.b,3)
            %有效误差矩阵
            res{i}.b(1:x,1:y,j) = kron(res{i+1}.b(:,:,j) , A);
        end
        %计算卷积层的核函数梯度
        if strcmp(cur,'Conv')
            %Initialize Conv Kernel
            res{i}.w = zeros(size(model.Layer{i}.k));
            
            for j = 1 : size(res{i}.w,3)
           %     res{i}.w(:,:,j) = conv2(data{i-1}(:,:,))
            end
            
        end
    end
    
    if strcmp(cur,'ANN')
        res{i}.b = model.Layer{i+1}.w'*res{i+1}.b.*(data{i}.*(1-data{i}));
        res{i}.w = res{i}.b*data{i-1}';
    end
end

end

