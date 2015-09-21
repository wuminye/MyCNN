addpath('./MNIST/');
addpath('./Core/');
addpath('./Util/');
model = InitCNNModel();
save model model;

%load model;

load picdata
num_train = 220;
alpha = 0.15;
flag = 1;
theta = SaveNetTheta(model);
J=[];
cor=[];
i = 1;
while flag ==1
    index = randperm(size(images,4),num_train);
    
    model = DropoutStart(model);
    
     F=@(p)CostFunction( p, images(:,:,:,index) , labels(index,:), model );
     options = optimset('MaxIter', 2);
     [theta, tt , model ,corind,pp] = fmincg(model,F, theta, options);
     J(end+1) = tt(end);
     cor(end+1) = pp(end);
  %  [ J(end+1), grad ,cor(end+1) ,~ ] = CostFunction( theta , images(:,:,:,index) , labels(index,:), model );
    model = DropoutEnd(model);
    
    %theta =  theta - alpha*grad;
    fprintf('%d : %e  %.4f\n',i,J(end),cor(end));
    i = i+1;
    if (mod(i,70)==0)
            num_train = floor(num_train * 1.1);
            fprintf('slow down\n');
    end
    if (i==2001)
        flag = 0;
    end
    model = LoadNetTheta(theta,model);
    save model model;
end
model = LoadNetTheta(theta,model);
save dataT J cor;
save model model;