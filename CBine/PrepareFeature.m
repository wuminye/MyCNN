function [ X ,CNT] = PrepareFeature( models, images)

N = size(images,4);
CNT = 0;
for i = 1:length(models)
    CNT = CNT + models{i}.sublayer{end}.subnet{end}.model.Layer{end-1}.out(1);
end

X = zeros(1,1,CNT,N);

parfor i = 1:N
    tem = [];
    for k = 1: length(models)
       res = cnnCalcForward(models{k},images(:,:,:,i)); 
       tem = [tem;res{end}{end}{end-1}(:)];
    end
    X(:,:,:,i) = reshape(tem,1,1,[]);
end

end

