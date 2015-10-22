function [ res ] = cnnSubNetGrad( model, data ,errdata,M,OnTrain)
%model 为模型，data为前向过程的计算结果
%最后一层的误差项直接由参数传入errdata ,一个struct 包含计算好的梯度
num = length(model.Layer);
res = cell(num,1);

res{num} = errdata;

for i = num-1:-1: 1
    t = model.Layer{i};
    cur = t.type;
    nex =  model.Layer{i+1}.type;
    res{i}.b = 0;
    res{i}.t = 0;
    res{i}.w = 0;
    
%=====================误差矩阵计算=================================
    if strcmp(nex,'Reshape')
       res{i}.t = reshape(res{i+1}.t,model.Layer{i}.out);
    end
    
    if strcmp(nex,'Conv')
        res{i}.t = zeros(model.Layer{i}.out);
       
      
        
        for p = 1:size(res{i}.t,3)
            for q = 1:size(res{i+1}.t,3)
                
              % ahpla = exp(model.Layer{i+1}.beta(p,q))/ sum(exp(model.Layer{i+1}.beta(:,q)));
               ahpla = 1;
               if model.Layer{i+1}.connector(q,p)~=1
                   continue;
               end
               res{i}.t(:,:,p) = res{i}.t(:,:,p) + ...
                        conv2(ahpla*res{i+1}.t(:,:,q),model.Layer{i+1}.w(:,:,p,q),'full');
            end
        end  
    end
    
     if strcmp(nex,'Convs')
        res{i}.t = zeros(model.Layer{i}.out);
       
        for p = 1:size(res{i}.t,3)
            for q = 1:size(res{i+1}.t,3)
               if model.Layer{i+1}.connector(q,p)~=1
                   continue;
               end
               tem = zeros(model.Layer{i}.out(1:2));
               tem(1:model.Layer{i+1}.stride:end,1:model.Layer{i+1}.stride:end)...
                   = res{i+1}.t(:,:,q);
               
               res{i}.t(:,:,p) = res{i}.t(:,:,p) + ...
                        conv2(tem,model.Layer{i+1}.w(:,:,p,q),'same');
            end
        end  
    end
    
    if strcmp(nex,'SoftMax')
        res{i}.t = (model.Layer{i+1}.w)'*reshape(res{i+1}.t,[],1); %.*(te.*(1-te));
        res{i}.t = reshape(res{i}.t,1,1,[]);
    end
    if  strcmp(nex,'ANN') 
        %te = reshape(data{i}(1,1,:), [] ,1);
        if  model.Layer{i+1}.dropout.enable == 1 && OnTrain == 1
             res{i}.t = (model.Layer{i+1}.w/ model.Layer{i+1}.dropout.p)'*reshape(res{i+1}.t.*reshape(model.Layer{i+1}.mask,1,1,[]),[],1); %.*(te.*(1-te));
         else 
             res{i}.t = (model.Layer{i+1}.w)'*reshape(res{i+1}.t,[],1); 
         end
        res{i}.t = reshape(res{i}.t,1,1,[]);
       
    end
    
    if strcmp(nex,'Pooling')
        k = model.Layer{i+1}.kernel;
        B = ones(k.x,k.y);
        %初始化误差featuremap
        res{i}.t = zeros(model.Layer{i}.out);
        %计算有效误差矩阵的大小
        x = size(res{i+1}.t,1)*k.x;
        y = size(res{i+1}.t,2)*k.y;
        
        for j = 1 : size(res{i+1}.t,3)
            %有效误差矩阵
            res{i}.t(1:x,1:y,j) = kron(res{i+1}.t(:,:,j) , B).*model.Layer{i+1}.w(j)./(k.x*k.y);
            %res{i}.t(1:x,1:y,j) = kron(res{i+1}.b(:,:,j) , B);
        end
        
    end
    
    if strcmp(nex,'MaxPooling')
        k = model.Layer{i+1}.kernel;
        B = ones(k.x,k.y);
        %初始化误差featuremap
        res{i}.t = zeros(model.Layer{i}.out);
        %计算有效误差矩阵的大小
        x = size(res{i+1}.t,1)*k.x;
        y = size(res{i+1}.t,2)*k.y;
        
        for j = 1 : size(res{i+1}.t,3)
            %有效误差矩阵
            res{i}.t(1:x,1:y,j) = kron(res{i+1}.t(:,:,j) , B);
            res{i}.t(1:x,1:y,j) = res{i}.t(1:x,1:y,j).*M{i+1}(:,:,j);
            %res{i}.t(1:x,1:y,j) = kron(res{i+1}.b(:,:,j) , B);
        end
        
    end
    
    %Input层不用计算梯度
    if i == 1
        break;
    end
%================梯度计算================================================    
     %计算卷积层的核函数梯度
    if strcmp(cur,'Conv') 

        %修正卷积层的误差，要乘以导数。
        res{i}.t = res{i}.t.*deActiveFunction(data{i});
        
        res{i}.w = zeros(size(model.Layer{i}.w));
        res{i}.b = zeros(size(model.Layer{i}.b));
        res{i}.beta = zeros(size(res{i}.w,3),size(res{i}.w,4));
        
      % ahpla = zeros(size(res{i}.w,3), size(res{i}.w,4));
      % tmp = zeros(size(res{i}.w,3), size(res{i}.w,4));
        for q = 1 : size(res{i}.w,4)
           for p = 1 : size(res{i}.w,3)
              % ahpla = 1;
               
              % ahpla(p,q) = exp(model.Layer{i}.beta(p,q))/ sum(exp(model.Layer{i}.beta(:,q)));
               
             %  tmp1 = res{i}.t(:,:,q).*conv2(data{i-1}(:,:,p),rot90(model.Layer{i}.w(:,:,p,q),2),'valid');
              % tmp(p,q) = sum(tmp1(:));
               %{
               for k = 1 : size(res{i}.w,3)
                  aa = - ahpla * exp(model.Layer{i}.beta(k,q))/ sum(exp(model.Layer{i}.beta(:,q)));
                  if k == p
                      aa = aa + ahpla;
                  end
                  res{i}.beta(p,q) = res{i}.beta(p,q) + aa*tmp1 ;
               end 
               %}
               
               if model.Layer{i}.connector(q,p)~=1
                   continue;
               end
              % res{i}.w(:,:,p,q) = conv2(data{i-1}(:,:,p), rot90(ahpla(p,q)*res{i}.t(:,:,q),2),'valid');
               res{i}.w(:,:,p,q) = conv2(data{i-1}(:,:,p), rot90(res{i}.t(:,:,q),2),'valid');
     
            end
            tem =res{i}.t(:,:,q);
            res{i}.b(q) = sum(tem(:));
        end
        %{
        for q = 1 : size(res{i}.w,4)
           for p = 1 : size(res{i}.w,3)
               pp = ahpla(:,q).*tmp(:,q);
               res{i}.beta(p,q) = ahpla(p,q)*(tmp(p,q) - sum(pp(:)));
           end
        end
        
       %}
        
    end
    
     %计算卷积层的核函数梯度
    if strcmp(cur,'Convs') 

        %修正卷积层的误差，要乘以导数。
        res{i}.t = res{i}.t.*deActiveFunction(data{i});
        
        res{i}.w = zeros(size(model.Layer{i}.w));
        res{i}.b = zeros(size(model.Layer{i}.b));
        for q = 1 : size(res{i}.w,4)
           for p = 1 : size(res{i}.w,3)
               if model.Layer{i}.connector(q,p)~=1
                   continue;
               end
               tem = zeros(model.Layer{i-1}.out(1:2));
               tem(1:model.Layer{i}.stride:end,1:model.Layer{i}.stride:end)...
                   = res{i}.t(:,:,q);
              
               
               %扩充矩阵!!!! stride为偶数  w的长宽 为奇数 
               [x1, y1] = size( data{i-1}(:,:,p));
                hk = floor(size(model.Layer{i}.w)/2);
               nf = zeros(size(tem,1)+hk(1)*2,...
                          size(tem,2)+hk(2)*2);
               nf(hk(1)+1:hk(1)+x1,hk(2)+1:hk(2)+y1) = data{i-1}(:,:,p);
               res{i}.w(:,:,p,q) = conv2(nf, rot90(tem,2),'valid');
   
            end
            tem =res{i}.t(:,:,q);
            res{i}.b(q) = sum(tem(:));
        end
        
    end
       
    if strcmp(cur,'Pooling') 
      %  res{i}.t = res{i}.t.*deActiveFunction(data{i});
         k = model.Layer{i}.kernel;
        res{i}.b = zeros( size(res{i}.t,3),1);
        res{i}.w = zeros( size(res{i}.t,3),1);
        for j = 1 : size(res{i}.t,3)
            xx = res{i}.t(:,:,j);
            res{i}.b(j) = sum(xx(:));
            ww = ((data{i}(:,:,j)-model.Layer{i}.b(j))/model.Layer{i}.w(j)).*res{i}.t(:,:,j);
            res{i}.w(j) = sum(ww(:));
        end

    end
    
    if strcmp(cur,'MaxPooling')
        % nothing to do
    end
    
    if strcmp(cur,'ANN')
        res{i}.t = res{i}.t.*deActiveFunction(data{i}).*reshape(model.Layer{i}.mask,1,1,[]);
        res{i}.b = res{i}.t;
        res{i}.t = reshape(res{i}.t,1,1,[]); 
        if  model.Layer{i}.dropout.enable == 1 && OnTrain == 1
             res{i}.w = res{i}.b(:)*reshape(data{i-1}, [] ,1)'; 
         else 
             res{i}.w = res{i}.b(:)*reshape(data{i-1}, [] ,1)'; 
        end
        res{i}.b = res{i}.t;
    end
%============================================================================    
end

end

