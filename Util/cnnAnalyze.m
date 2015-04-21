function [ J , cor ,ind ,indf,model,tp,tn] = cnnAnalyze( model,num,images,labels )
% J ----  CostFuncion 的值
% cor  -------  正确率
% ind  -------  正确样本在images中的编号序列
% indf ---------- 错误样本在images中的编号序列
% tp ------   true positive
% tn ------   true negative


if ~exist('num', 'var')
    num = size(images,4);
end
tt = [1:size(images,4)]';

%===================================
%随机选取num个样本分析
index = randperm(size(images,4),num); 
images = images(:,:,:,index);
labels = labels(index,:);
tt = tt(index);
%===================================

lambda = model.lambda;

ind = zeros(num,1);
J = 0;
%计算每个样本的带价值和修正梯度
cor = 0;
tp = 0;
tn =0;
parfor i = 1 : num
   res = cnnCalcForward(model,images(:,:,:,i));
    output = res{end}{end}{end}(:);

    yy = labels(i,:)';
  if onehot2num(output) == onehot2num(yy)
       cor = cor + 1;
       ind(i)=1;
       if onehot2num(output) == 1
           tp = tp + 1;
       else
           tn = tn + 1;
       end
  end
%============代价函数计算==============
   %J = J + (-yy'*log(output)-(1-yy')*log(1-output) ); 
   
   %使用SoftMax的代价函数
   J = J + -yy'*log(output);
%====================================

end;
J = J / num;
J = J + lambda*cnnCalcNetReg(model)/(2*num);

te=ind;
ind = tt(logical(ind));
indf = tt(~logical(te));
cor = cor/num*100;



tp = tp / sum(sum(labels(:,1)==1));
tn = tn / sum(sum(labels(:,2)==1));
end

function r = onehot2num(y)
[a,r] = max(y);
end
