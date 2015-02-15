function [ J , cor ] = cnnAnalyze( model,num,images,labels )
%CNNANALYZE Summary of this function goes here
%   Detailed explanation goes here


if ~exist('num', 'var')
    num = size(images,4);
end

%随机选取num个样本分析
index = randperm(size(images,4),num); 

images = images(:,:,:,index);
labels = labels(index,:);


lambda = model.lambda;

J = 0;
%计算每个样本的带价值和修正梯度
cor = 0;
parfor i = 1 : num
   res = cnnCalcnet(model,images(:,:,:,i));
   output = res{length(res)}(:);
   %yy = zeros(size(output,1),1);
   %yy(labels(i)) = 1;
   yy = labels(i,:)';
  if onehot2num(output) == onehot2num(yy)
       cor = cor + 1;
  end
%============代价函数计算==============
   %J = J + (-yy'*log(output)-(1-yy')*log(1-output) ); 
   
   %使用SoftMax的代价函数
   J = J + -yy'*log(output);
%====================================

end;
J = J / num;
J = J + lambda*cnnCalcReg(model)/(2*num);

cor = cor/num*100;

end

function r = onehot2num(y)
[a,r] = max(y);
end
