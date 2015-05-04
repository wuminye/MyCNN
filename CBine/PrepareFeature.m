function [ X ] = PrepareFeature( models, images)

N = size(images,4);
res = cnnCalcForward(models{1},images(:,:,:,1));
output = res{end}{end}{end-1};

X = zeros(1,1,size(output,3)*length(models),N);

parfor i = 1:N
    tem = [];
    for k = 1: length(models)
       res = cnnCalcForward(models{k},images(:,:,:,i)); 
       tem = [tem;res{end}{end}{end-1}(:)];
    end
    X(:,:,:,i) = reshape(tem,1,1,[]);
end

end

