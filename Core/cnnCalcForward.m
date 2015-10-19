function [ res , inFeatureMap,M] = cnnCalcForward( model ,data ,state )
%������������ļ���
%data Ϊfeaturemap��ʽ
%model�ĸ�ʽ:   model.sublayer{i}.subnet{j} Ϊ��i���е�j��subnet
%  model.sublayer{i}.connect(q,p) Ϊsublayer i-1 (q) �� i (p) ֮������ӹ�ϵ��Ϊһ��0/1����
% if state = 1 then means the lastest subnet is going to be ignored.

if ~exist('state', 'var')
    state = 0;
end


num_sublayer = length(model.sublayer); %����м���.
res = cell(num_sublayer,1);
M = cell(num_sublayer,1);
res{1}{1}{1} = data;



for i = 2 : num_sublayer
    
    num_subnet = length(model.sublayer{i}.subnet); %��ȡ��ǰ��subnet�ĸ���
    res{i} = cell(num_subnet,1);
    M{i} = cell(num_subnet,1);
    for j = 1:num_subnet 
       
        %---------------------------------------------------
        %�������һ��sublayer��������������ɺϲ������������featureMAP
        
        %����һsublayer ��subnet��������Ӹ����� 
        n_con = length(model.sublayer{i-1}.subnet);     

        
        LastOutSize = size(res{i-1}{1}{end}); %��һ��feature�ĳߴ�
        for k = 2:length(model.sublayer{i-1}.subnet)
            LastOutSize = min([LastOutSize ; size(res{i-1}{k}{end})]);
        end
        
        for k = 1:length(model.sublayer{i-1}.subnet)
            res{i-1}{k}{end} = res{i-1}{k}{end}(1:LastOutSize(1),1:LastOutSize(2),:);
        end
        
        
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
        
        %ֻҪ������һ��subnet��inFeatureMap
        if   i == num_sublayer  &&   state == 1
            break;
        end
        
        [res{i}{j},M{i}{j}] = ...
            cnnCalcSubnet( model.sublayer{i}.subnet{j}.model ,inFeatureMap ,model.OnTrain);

    end
   
end;



end

