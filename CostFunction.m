function [ J, grad ] = CostFunction( theta , input , y, model ,lambda)

num_data = size(input,4);
model = LoadTheta(theta,model);
J = 0;
res = cell(num_data,1);
T   = cell(num_data,1);

%计算每个样本的带价值和修正梯度
cor = 0;
for i = 1 : num_data
   res{i} = cnnCalcnet(model,input(:,:,:,i));
   output = res{i}{length(res{i})}(:);
   yy = zeros(size(output,1),1);
   yy(y(i)) = 1;
   [q,ar] = max(output);
   if ar == y(i)
       cor = cor + 1;
   end
   J = J + (-yy'*log(output)-(1-yy')*log(1-output) ); 
   T{i} = cnnGrad( model, res{i} , yy ,num_data);
   if mod(i,4000)==0
      %fprintf('.');
    end
end;
J = J / num_data;

J = J + lambda*cnnCalcReg(model)/(2*num_data);

grad = [];
fs = T{1};
% 合并多个样本的梯度
for i = 2 : num_data
    for j = 1: length(T{i})
       if strcmp(model.Layer{j}.type,'ANN') || strcmp(model.Layer{j}.type,'Conv')
           fs{j}.w = fs{j}.w + T{i}{j}.w;
           fs{j}.b = fs{j}.b + T{i}{j}.b;
       end
    end
end
%fprintf('%.5f%% \n',cor/num_data*100);
%计算正则项梯度偏差
for j = 1: length(model.Layer)
   if strcmp(model.Layer{j}.type,'ANN') || strcmp(model.Layer{j}.type,'Conv')
      fs{j}.w = fs{j}.w + lambda*model.Layer{j}.w./num_data;
   end
end

%生成梯度向量
for i = 1 : length(fs)
    if strcmp(model.Layer{i}.type,'ANN') || strcmp(model.Layer{i}.type,'Conv')
        grad = [grad;fs{i}.b(:);fs{i}.w(:);];
    end
end
%fprintf('epoch over.\n');
end

