function [ J , cor ] = cnnAnalyze( model,num,images,labels )
%CNNANALYZE Summary of this function goes here
%   Detailed explanation goes here


if ~exist('num', 'var')
    num = size(images,4);
end

%���ѡȡnum����������
index = randperm(size(images,4),num); 

images = images(:,:,:,index);
labels = labels(index,:);


lambda = model.lambda;

J = 0;
%����ÿ�������Ĵ���ֵ�������ݶ�
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
%============���ۺ�������==============
   %J = J + (-yy'*log(output)-(1-yy')*log(1-output) ); 
   
   %ʹ��SoftMax�Ĵ��ۺ���
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
