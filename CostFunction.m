function [ J, grad ,cor ,corind ] = CostFunction( theta , input , y, model )
lambda = model.lambda;
num_data = size(input,4);
model = LoadTheta(theta,model);
J = 0;

T   = cell(num_data,1);

%��ȷ��ʶ
corind = zeros(num_data,1);

%����ÿ�������Ĵ���ֵ�������ݶ�
cor = 0;

parfor i = 1 : num_data
   res = cnnCalcnet(model,input(:,:,:,i));
   output = res{length(res)}(:);
   %yy = zeros(size(output,1),1);
   %yy(y(i)) = 1;
   yy = y(i,:)';
  
   if onehot2num(output) == onehot2num(y(i,:))
       cor = cor + 1;
       corind(i) = 1; %��¼��ȷ��ʶ
   end
%============���ۺ�������==============
   %J = J + (-yy'*log(output)-(1-yy')*log(1-output) ); 
   
   %ʹ��SoftMax�Ĵ��ۺ���
   J = J + -yy'*log(output);
%====================================
   T{i} = cnnGrad( model, res , yy ,num_data);
 %  if mod(i,4000)==0
      %fprintf('.');
  % end
 
end;
J = J / num_data;

J = J + lambda*cnnCalcReg(model)/(2*num_data);

grad = [];
fs = T{1};
% �ϲ�����������ݶ�
for i = 2 : num_data
    for j = 1: length(T{i})
       if strcmp(model.Layer{j}.type,'ANN') || strcmp(model.Layer{j}.type,'Conv')
           fs{j}.w = fs{j}.w + T{i}{j}.w;
           fs{j}.b = fs{j}.b + T{i}{j}.b;
       end
       if strcmp(model.Layer{j}.type,'SoftMax')
           fs{j}.w = fs{j}.w + T{i}{j}.w;
       end
    end
end
cor = cor/num_data*100;
fprintf('%.5f%%  %e \r',cor,J);
%�����������ݶ�ƫ��
for j = 1: length(model.Layer)
   if strcmp(model.Layer{j}.type,'ANN') || strcmp(model.Layer{j}.type,'Conv') ...
           || strcmp(model.Layer{j}.type,'SoftMax')
      fs{j}.w = fs{j}.w + lambda*model.Layer{j}.w./num_data;
   end
end

%�����ݶ�����
for i = 1 : length(fs)
    if strcmp(model.Layer{i}.type,'ANN') || strcmp(model.Layer{i}.type,'Conv') 
        grad = [grad;fs{i}.b(:);fs{i}.w(:);];
    end
    if strcmp(model.Layer{i}.type,'SoftMax')
        grad = [grad;fs{i}.w(:);];
    end
end
%fprintf('epoch over.\n');
end


function r = onehot2num(y)
[a,r] = max(y);
end
