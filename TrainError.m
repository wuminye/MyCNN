addpath('./MNIST/');
addpath('./Core/');
addpath('./Util/');

load model
load picdata;


lambda = model.lambda;
num = size(images,4);
 fprintf('%d\n',num);
ind = zeros(num,1);
J=0;
cor = 0;
parfor i = 1 : num
   res = cnnCalcForward(model,images(:,:,:,i));
    output = res{end}{end}{end}(:);

    yy = labels(i,:)';
    
    [~,r1] = max(output);
    [~,r2] = max(yy);
  if r1 ~= r2
       cor = cor + 1;
       ind(i)=1;
  end
%============代价函数计算==============
  
   %使用SoftMax的代价函数
   J = J + -yy'*log(output);
%====================================
    fprintf('\r%d',i);
end;
J = J / num;
J = J + lambda*cnnCalcNetReg(model)/(2*num);


fprintf('number of Error data:%d     J: %e\n',cor,J);

ind(randperm(size(ind,1),sum(ind(:))*2)) =1;

ind = logical(ind);



images = images(:,:,:,ind);
labels = labels(ind,:);

itn = input('the number of iteration : ');
dropout = input('Enable Dropout? : ');
tic;
theta = SaveNetTheta(model);
options = optimset('MaxIter', itn);
F=@(p)CostFunction( p, images, labels, model );

  if dropout == 1
   model = DropoutStart(model);
  end
  [nn_params, cost , model ,corind] = fmincg(model,F, theta, options);
  if dropout == 1
     model = DropoutEnd(model);
  end
  
  
 model = LoadNetTheta(nn_params,model);
 save model  model;
 toc;