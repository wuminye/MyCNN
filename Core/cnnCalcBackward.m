function [ res ] = cnnCalcBackward( model, data , errdata )

num_sublayer = length(model.sublayer); %����м���.
res = cell(num_sublayer,1);

for i = num_sublayer:-1: 2
 
     num_subnet = length(model.sublayer{i}.subnet); %��ȡ��ǰ��subnet�ĸ���
     res{i} = cell(num_subnet,1);
     
     %����ÿ��subnet�������featuremap�ĸ���
     output_featuremap_num = zeros(num_subnet,1);
     for j = 1 : num_subnet
         output_featuremap_num(j) = size(data{i}{j}{end},3);
     end
     
     
     for j = 1 : num_subnet
        %  res{i}{j} = cell(length(model.sublayer{i}.subnet{j}.model.Layer),1);
          %���һ���errdata�ɲ�������
          theta = errdata;
          if i ~= num_sublayer              
             %����һsublayer ��subnet��������Ӹ����� ��Ҫճ�ϼ�
             n_con = length(model.sublayer{i+1}.subnet);  
             
             theta.t = zeros(size(data{i}{j}{end}));
             
             con =  model.sublayer{i+1}.connect(j,:);
             
             %����������
             for k = 1 : n_con
                if (con(1,k)==0) 
                    continue;
                end
                
                bcon =  model.sublayer{i+1}.connect(1:j-1,k) ;
                
                %���featuremap�Ŀ�ʼ���
                be = sum(output_featuremap_num(logical(bcon)))+1; 
                
                tem =  res{i+1}{k}{1}.t(:,:,be:be+size(theta.t,3)-1);
                theta.t = theta.t + tem;
             end
             
             
             %---------�����ݶ�-------------------------
                model_layer = model.sublayer{i}.subnet{j}.model.Layer{end};
                cur = model_layer.type;
               
             
                  %��������ĺ˺����ݶ�
                if strcmp(cur,'Conv') 

                    %������������Ҫ���Ե�����
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

                 %��������ĺ˺����ݶ�
                if strcmp(cur,'Convs') 

                    %������������Ҫ���Ե�����
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


                           %�������!!!! strideΪż��  w�ĳ��� Ϊ���� 
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

                   % res{i}.b = res{i}.t ; %.*(data{i}.*(1-data{i}));Pooling��û�м����

                end


                if strcmp(cur,'ANN')
                   theta.t = theta.t.*deActiveFunction( data{i}{j}{end});
                    theta.b = theta.t;
                    theta.t = reshape(theta.t,1,1,[]); % ��Ҫ�ɲ�Ҫ����
                    theta.w = theta.b(:)*reshape(data{i}{j}{end-1}, [] ,1)';
                    theta.b = theta.t;
                end
             
          end
          
          res{i}{j} = cnnSubNetGrad(model.sublayer{i}.subnet{j}.model,...
                                    data{i}{j},theta);
          
          
     end
    
    
end    
    
    
end


