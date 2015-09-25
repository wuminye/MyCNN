function [ J, grad ,cor ,corind ] = CostFunction( theta , input , y, model )
lambda = model.lambda;
num_data = size(input,4);
model = LoadNetTheta(theta,model);
J = 0;

T   = cell(num_data,1);

%正确标识
corind = zeros(num_data,1);

%计算每个样本的带价值和修正梯度
cor = 0;

parfor i = 1 : num_data
   res = cnnCalcForward(model,input(:,:,:,i));
   output = res{end}{end}{end}(:);
   %yy = zeros(size(output,1),1);
   %yy(y(i)) = 1;
   yy = y(i,:)';
  
   if onehot2num(output) == onehot2num(y(i,:))
       cor = cor + 1;
       corind(i) = 1; %记录正确标识
   end
%============代价函数计算==============
   %J = J + (-yy'*log(output)-(1-yy')*log(1-output) ); 
   
   %使用SoftMax的代价函数
   J = J + -yy'*log(output);
%====================================

%============================================
   %计算softmax的误差了梯度，生成errdata
   te1 = res{end}{end}{end}(1,1,:);
   te2 = res{end}{end}{end-1}(1,1,:);
   te1 = te1(:);te2 = te2(:);

   a = reshape((te1 - yy)./num_data,1,1,[]);
   b = (te1 - yy)*te2'./num_data;

   T{i} = cnnCalcBackward( model, res , makestruct(a,b) );

 
%=============================================== 
end;


J = J / num_data;
%加入正则项
J = J + lambda*cnnCalcNetReg(model)/(2*num_data);

grad = [];
fs = T{1};
% 合并多个样本的梯度
for i = 2 : num_data
    fs = mergeGrad(fs,T{i},model);
end

cor = cor/num_data*100;
%fprintf('%.5f%%  %e \r',cor,J);


%计算正则项梯度偏差

fs = addGradReg(fs,model ,num_data);

%生成梯度向量
grad = genGrad(fs,model);
%fprintf('epoch over.\n');
end


function  grad1 = mergeGrad(grad1 , grad2 , mo)

   for i = 1: length(grad1)
       for j = 1: length(grad1{i})
           model = mo.sublayer{i}.subnet{j}.model;
           for k = 1: length(grad1{i}{j})
               if strcmp(model.Layer{k}.type,'ANN') || strcmp(model.Layer{k}.type,'Conv') ...
                       || strcmp(model.Layer{k}.type,'Convs')
                   grad1{i}{j}{k}.w = grad1{i}{j}{k}.w + grad2{i}{j}{k}.w;
                   grad1{i}{j}{k}.b = grad1{i}{j}{k}.b + grad2{i}{j}{k}.b;
               end
               if strcmp(model.Layer{k}.type,'SoftMax')
                   grad1{i}{j}{k}.w = grad1{i}{j}{k}.w + grad2{i}{j}{k}.w;
               end
               
                if strcmp(model.Layer{k}.type,'Pooling')
                   grad1{i}{j}{k}.b = grad1{i}{j}{k}.b + grad2{i}{j}{k}.b;
                   grad1{i}{j}{k}.w = grad1{i}{j}{k}.w + grad2{i}{j}{k}.w;
                end
               
           end
       end
    end

end


function  grad = addGradReg(grad,mo ,num_data)

   for i = 1: length(grad)
       for j = 1: length(grad{i})
           model = mo.sublayer{i}.subnet{j}.model;
           for k = 1: length(grad{i}{j})
               if strcmp(model.Layer{k}.type,'Convs')
                   grad{i}{j}{k}.w = grad{i}{j}{k}.w + mo.lambda*model.Layer{k}.w./num_data;
               end
               if strcmp(model.Layer{k}.type,'ANN')  
                   tmp = model.Layer{k}.w;
                   tmp(model.Layer{k}.w>0) = 1;
                   tmp(model.Layer{k}.w<0) = -1;
                   grad{i}{j}{k}.w = grad{i}{j}{k}.w + 0.5*mo.lambda*tmp./num_data;
               end
               if  strcmp(model.Layer{k}.type,'Conv')
                   grad{i}{j}{k}.w = grad{i}{j}{k}.w + mo.lambda*model.Layer{k}.w./num_data;
                   grad{i}{j}{k}.beta = grad{i}{j}{k}.beta + mo.lambda*model.Layer{k}.beta./num_data;
               end
               if  strcmp(model.Layer{k}.type,'Pooling')
                   grad{i}{j}{k}.w = grad{i}{j}{k}.w + mo.lambda*model.Layer{k}.w./num_data;
               end
               if strcmp(model.Layer{k}.type,'SoftMax')
                    tmp = model.Layer{k}.w;
                   tmp(model.Layer{k}.w>0) = 1;
                   tmp(model.Layer{k}.w<0) = -1;
                     grad{i}{j}{k}.w = grad{i}{j}{k}.w + 0.5*mo.lambda*tmp./num_data;
               end
                             
           end
       end
    end

end

function  dgrad = genGrad(grad,mo)
   dgrad = [];
   for i = 1: length(grad)
       for j = 1: length(grad{i})
           model = mo.sublayer{i}.subnet{j}.model;
           for k = 1: length(grad{i}{j})
               if strcmp(model.Layer{k}.type,'ANN')  ...
                       || strcmp(model.Layer{k}.type,'Convs')
                   dgrad = [dgrad; grad{i}{j}{k}.b(:); grad{i}{j}{k}.w(:);];
     
               end
               
               if strcmp(model.Layer{k}.type,'Conv')
                  dgrad = [dgrad; grad{i}{j}{k}.b(:); grad{i}{j}{k}.w(:);grad{i}{j}{k}.beta(:);];
                %  dgrad = [dgrad; grad{i}{j}{k}.b(:); grad{i}{j}{k}.w(:);];
               end 
               
               
               if strcmp(model.Layer{k}.type,'SoftMax')
                   dgrad = [dgrad;grad{i}{j}{k}.w(:);];
               end
               
                if strcmp(model.Layer{k}.type,'Pooling')
                   dgrad = [dgrad;grad{i}{j}{k}.b(:); grad{i}{j}{k}.w(:);];
               end
               
           end
       end
    end

end

function errdata = makestruct(a,b)
     errdata.t = a;
     errdata.w = b;
  
end

function r = onehot2num(y)
[a,r] = max(y);
end
