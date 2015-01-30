function [ res ] = cnnGrad( model, data , y ,m )
%CNNGRAD Summary of this function goes here
%   Detailed explanation goes here
num = length(model.Layer);
res = cell(num,1);

%���һ������ֵ,Ҫ��֤������㶼��һά��
te1 = data{num}(1,1,:);
te2 = data{num-1}(1,1,:);
te1 = te1(:);te2 = te2(:);
if strcmp(model.Layer{num}.type,'ANN')
%---------logic regression --------
res{num}.t = reshape((te1(:) - y)./m,1,1,[]);
res{num}.b = res{num}.t;
res{num}.w = (te1(:) - y)*te2(:)'./m;
%----------------------------------
else
%---------SoftMax regression-------
res{num}.t = reshape((te1 - y)./m,1,1,[]);
res{num}.w = (te1 - y)*te2'./m;

%----------------------------------
end

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
        res{i}.t = res{i}.t.*(data{i}.*(1-data{i}));
        
        res{i}.w = zeros(size(model.Layer{i}.w));
        res{i}.b = zeros(size(model.Layer{i}.b));
        for q = 1 : size(res{i}.w,4)
           for p = 1 : size(res{i}.w,3)
               res{i}.w(:,:,p,q) = conv2(data{i-1}(:,:,p), rot90(rot90(res{i}.t(:,:,q))),'valid');
               %------------����Ҫ��Ҫ�ٷ�ת������?????????????????????
               %res{i}.w(:,:,p,q) = rot90(rot90(conv2(data{i-1}(:,:,p), rot90(rot90(res{i}.t(:,:,q))),'valid')));
            end
            tem =res{i}.t(:,:,q);
            res{i}.b(q) = sum(tem(:));
        end
        
    end
       
    if strcmp(cur,'Pooling') 
 
       % res{i}.b = res{i}.t ; %.*(data{i}.*(1-data{i}));Pooling��û�м����

    end
    
    
    if strcmp(cur,'ANN')
        res{i}.t = res{i}.t.*(data{i}.*(1-data{i}));
        res{i}.b = res{i}.t;
        res{i}.t = reshape(res{i}.t,1,1,[]); % ��Ҫ�ɲ�Ҫ����
        res{i}.w = res{i}.b(:)*reshape(data{i-1}, [] ,1)';
        res{i}.b = res{i}.t;
    end
%============================================================================    
end

end

