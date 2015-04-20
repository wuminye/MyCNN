function [ res ] = cnnCalcBackward( model, data , errdata )

num_sublayer = length(model.sublayer); %获得有几层.
res = cell(num_sublayer,1);

for i = num_sublayer:-1: 2
 
     num_subnet = length(model.sublayer{i}.subnet); %获取当前层subnet的个数
     res{i} = cell(num_subnet,1);
     
     %计算每个subnet的输出的featuremap的个数
     output_featuremap_num = zeros(num_subnet,1);
     for j = 1 : num_subnet
         output_featuremap_num(j) = size(data{i}{j}{end},3);
     end
     
     
     for j = 1 : num_subnet
        %  res{i}{j} = cell(length(model.sublayer{i}.subnet{j}.model.Layer),1);
          %最后一层的errdata由参数传入
          theta = errdata;
          if i ~= num_sublayer              
             %与下一sublayer 的subnet的最多连接个数； 需要粘合剂
             n_con = length(model.sublayer{i+1}.subnet);  
             
             theta.t = zeros(size(data{i}{j}{end}));
             
             con =  model.sublayer{i+1}.connect(j,:);
             
             %计算误差矩阵
             for k = 1 : n_con
                if (con(1,k)==0) 
                    continue;
                end
                
                bcon =  model.sublayer{i+1}.connect(1:j-1,k) ;
                
                %获得featuremap的开始编号
                be = sum(output_featuremap_num(logical(bcon)))+1; 
                
                tem =  res{i+1}{k}{1}.t(:,:,be:be+size(theta.t,3)-1);
                theta.t = theta.t + tem;
             end
             
             
             %---------计算梯度-------------------------
                model_layer = model.sublayer{i}.subnet{j}.model.Layer{end};
                cur = model_layer.type;
               
             
                  %计算卷积层的核函数梯度
                if strcmp(cur,'Conv') 

                    %修正卷积层的误差，要乘以导数。
                    theta.t = theta.t.*deActiveFunction(data{i}{j}{end});

                    theta.w = zeros(size(model_layer.w));
                    theta.b = zeros(size(model_layer.b));
                    for q = 1 : size( theta.w,4)
                       for p = 1 : size(theta.w,3)
                           if  model_layer.connector(q,p)~=1
                               continue;
                           end
                           theta.w(:,:,p,q) = ...
                               conv2(data{i}{j}{end-1}(:,:,p), rot90(rot90(theta.t(:,:,q))),'valid');

                        end
                        tem =theta.t(:,:,q);
                        theta.b(q) = sum(tem(:));
                    end

                end

                 %计算卷积层的核函数梯度
                if strcmp(cur,'Convs') 

                    %修正卷积层的误差，要乘以导数。
                   theta.t = theta.t.*deActiveFunction(data{i}{j}{end});

                    theta.w = zeros(size(model_layer.w));
                    theta.b = zeros(size(model_layer.b));
                    
                    for q = 1 : size( theta.w,4)
                       for p = 1 : size(theta.w,3)
                           if  model_layer.connector(q,p)~=1
                               continue;
                           end
                           tem = zeros( model.sublayer{i}.subnet{j}.model.Layer{end-1}.out(1:2));
                           tem(1:model_layer.stride:end,1:model_layer.stride:end)...
                               = theta.t(:,:,q);


                           %扩充矩阵!!!! stride为偶数  w的长宽 为奇数 
                           [x1, y1] = size( data{i}{j}{end-1}(:,:,p));
                            hk = floor(size(model_layer.w)/2);
                           nf = zeros(size(tem,1)+hk(1)*2,...
                                      size(tem,2)+hk(2)*2);
                           nf(hk(1)+1:hk(1)+x1,hk(2)+1:hk(2)+y1) = data{i}{j}{end-1}(:,:,p);
                           theta.w(:,:,p,q) = conv2(nf, rot90(tem,2),'vaild');

                        end
                        tem =theta.t(:,:,q);
                        theta.b(q) = sum(tem(:));
                    end

                end

                if strcmp(cur,'Pooling') 

                   % res{i}.b = res{i}.t ; %.*(data{i}.*(1-data{i}));Pooling层没有激活函数

                end


                if strcmp(cur,'ANN')
                   theta.t = theta.t.*deActiveFunction( data{i}{j}{end});
                    theta.b = theta.t;
                    theta.t = reshape(theta.t,1,1,[]); % 可要可不要？？
                    theta.w = theta.b(:)*reshape(data{i}{j}{end-1}, [] ,1)';
                    theta.b = theta.t;
                end
             
          end
          
          res{i}{j} = cnnSubNetGrad(model.sublayer{i}.subnet{j}.model,...
                                    data{i}{j},theta);
          
          
     end
    
    
end    
    
    
end


