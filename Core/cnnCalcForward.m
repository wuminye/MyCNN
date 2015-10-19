function [ res , inFeatureMap,M] = cnnCalcForward( model ,data ,state )
%对于整合网络的计算
%data 为featuremap格式
%model的格式:   model.sublayer{i}.subnet{j} 为第i层中第j个subnet
%  model.sublayer{i}.connect(q,p) 为sublayer i-1 (q) 与 i (p) 之间的连接关系，为一个0/1矩阵
% if state = 1 then means the lastest subnet is going to be ignored.

if ~exist('state', 'var')
    state = 0;
end


num_sublayer = length(model.sublayer); %获得有几层.
res = cell(num_sublayer,1);
M = cell(num_sublayer,1);
res{1}{1}{1} = data;



for i = 2 : num_sublayer
    
    num_subnet = length(model.sublayer{i}.subnet); %获取当前层subnet的个数
    res{i} = cell(num_subnet,1);
    M{i} = cell(num_subnet,1);
    for j = 1:num_subnet 
       
        %---------------------------------------------------
        %检查与上一个sublayer的连接情况，生成合并后的输入数据featureMAP
        
        %与上一sublayer 的subnet的最多连接个数； 
        n_con = length(model.sublayer{i-1}.subnet);     

        
        LastOutSize = size(res{i-1}{1}{end}); %上一层feature的尺寸
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
        
        %只要获得最后一个subnet的inFeatureMap
        if   i == num_sublayer  &&   state == 1
            break;
        end
        
        [res{i}{j},M{i}{j}] = ...
            cnnCalcSubnet( model.sublayer{i}.subnet{j}.model ,inFeatureMap ,model.OnTrain);

    end
   
end;



end

