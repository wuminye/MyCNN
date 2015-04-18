function [ res ] = cnnCalcForward( model ,data )
%������������ļ���
%data Ϊfeaturemap��ʽ
%model�ĸ�ʽ:   model.sublayer{i}.subnet{j} Ϊ��i���е�j��subnet
%  model.sublayer{i}.connect(q,p) Ϊsublayer i-1 (q) �� i (p) ֮������ӹ�ϵ��Ϊһ��0/1����


num_sublayer = length(model.sublayer); %����м���.
res = cell(num_sublayer,1);
res{1}{1}{1} = data;



for i = 2 : num_sublayer
    
    num_subnet = length(model.sublayer{i}.subnet); %��ȡ��ǰ��subnet�ĸ���
    res{i} = cell(num_subnet,1);
    for j = 1:num_subnet 
       
        %---------------------------------------------------
        %�������һ��sublayer��������������ɺϲ������������featureMAP
        
        %����һsublayer ��subnet��������Ӹ����� 
        n_con = length(model.sublayer{i-1}.subnet);     

        
        LastOutSize = size(res{i-1}{1}{end}); %��һ��feature�ĳߴ�
        LastOutSize(3) = 0;
        inFeatureMap = zeros(LastOutSize );
        
        for k = 1: n_con
            if model.sublayer{i}.connect(k,j) == 0
               continue; 
            end
           % inFeatureMap = mergeFeatureMap(inFeatureMap,res{i-1}{k}{end});
            
            tem = zeros(size(inFeatureMap,1),size(inFeatureMap,2),...
                   size(inFeatureMap,3)+size(res{i-1}{k}{end},3));
            tem(:,:,1:size(inFeatureMap,3)) =inFeatureMap;
            tem(:,:,size(inFeatureMap,3)+1:end)=res{i-1}{k}{end};
            inFeatureMap = tem;
            
        end
        
        %-------------OUT: inFeatureMap ---------------------
        
        res{i}{j} = ...
            cnnCalcSubnet( model.sublayer{i}.subnet{j}.model ,inFeatureMap );

    end
   
end;



end

