function [ res ] = cnnGrad( model, data ,errdata)
%model Ϊģ�ͣ�dataΪǰ����̵ļ�����
%���һ��������ֱ���ɲ�������errdata ,Ϊһ��struct
num = length(model.Layer);
res = cell(num,1);

res{num} = errdata;

for i = num-1:-1: 2
    t = model.Layer{i};
    cur = t.type;
    nex =  model.Layer{i+1}.type;
    res{i}.b = 0;
    res{i}.t = 0;
    res{i}.w = 0;
    
%=====================���������=================================
    if strcmp(nex,'Reshape')
       res{i}.t = reshape(res{i+1}.t,model.Layer{i}.out);
    end
    
    if strcmp(nex,'Conv')
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
    
    
    if  strcmp(nex,'ANN') || strcmp(nex,'SoftMax')
        %te = reshape(data{i}(1,1,:), [] ,1);
        res{i}.t = model.Layer{i+1}.w'*reshape(res{i+1}.t,[],1); %.*(te.*(1-te));
        res{i}.t = reshape(res{i}.t,1,1,[]);
    end
    
    if strcmp(nex,'Pooling')
        k = model.Layer{i+1}.kernel;
        B = ones(k.x,k.y);
        %��ʼ�����featuremap
        res{i}.t = zeros(model.Layer{i}.out);
        %������Ч������Ĵ�С
        x = size(res{i+1}.t,1)*k.x;
        y = size(res{i+1}.t,2)*k.y;
        
        for j = 1 : size(res{i+1}.t,3)
            %��Ч������
            res{i}.t(1:x,1:y,j) = kron(res{i+1}.t(:,:,j) , B)./(k.x*k.y);
            %res{i}.t(1:x,1:y,j) = kron(res{i+1}.b(:,:,j) , B);
        end
        
    end
    
%================�ݶȼ���================================================    
     %��������ĺ˺����ݶ�
    if strcmp(cur,'Conv') 

        %������������Ҫ���Ե�����
        res{i}.t = res{i}.t.*deActiveFunction(data{i});
        
        res{i}.w = zeros(size(model.Layer{i}.w));
        res{i}.b = zeros(size(model.Layer{i}.b));
        for q = 1 : size(res{i}.w,4)
           for p = 1 : size(res{i}.w,3)
               if model.Layer{i}.connector(q,p)~=1
                   continue;
               end
               res{i}.w(:,:,p,q) = conv2(data{i-1}(:,:,p), rot90(rot90(res{i}.t(:,:,q))),'valid');
               %------------����Ҫ��Ҫ�ٷ�ת������?????????????????????
               %res{i}.w(:,:,p,q) = rot90(rot90(conv2(data{i-1}(:,:,p), rot90(rot90(res{i}.t(:,:,q))),'valid')));
            end
            tem =res{i}.t(:,:,q);
            res{i}.b(q) = sum(tem(:));
        end
        
    end
    
     %��������ĺ˺����ݶ�
    if strcmp(cur,'Convs') 

        %������������Ҫ���Ե�����
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
              
               
               %�������!!!! strideΪż��  w�ĳ��� Ϊ���� 
               [x1, y1] = size( data{i-1}(:,:,p));
                hk = floor(size(model.Layer{i}.w)/2);
               nf = zeros(size(tem,1)+hk(1)*2,...
                          size(tem,2)+hk(2)*2);
               nf(hk(1)+1:hk(1)+x1,hk(2)+1:hk(2)+y1) = data{i-1}(:,:,p);
               res{i}.w(:,:,p,q) = conv2(nf, rot90(tem,2),'vaild');
   
            end
            tem =res{i}.t(:,:,q);
            res{i}.b(q) = sum(tem(:));
        end
        
    end
       
    if strcmp(cur,'Pooling') 
 
       % res{i}.b = res{i}.t ; %.*(data{i}.*(1-data{i}));Pooling��û�м����

    end
    
    
    if strcmp(cur,'ANN')
        res{i}.t = res{i}.t.*deActiveFunction(data{i});
        res{i}.b = res{i}.t;
        res{i}.t = reshape(res{i}.t,1,1,[]); % ��Ҫ�ɲ�Ҫ����
        res{i}.w = res{i}.b(:)*reshape(data{i-1}, [] ,1)';
        res{i}.b = res{i}.t;
    end
%============================================================================    
end

end

