function [ X, y ] = LoadData( name )
%LOADDATA Summary of this function goes here
%   Detailed explanation goes here
if strcmp(name,'MNIST')
   imageDim = 28;
   images = loadMNISTImages('train-images.idx3-ubyte');
   images = reshape(images,imageDim,imageDim,1,[]);
   X = images;
   labels = loadMNISTLabels('train-labels.idx1-ubyte');
   labels(labels==0) =10;
   y = zeros(size(labels,1),10);
    for p = 1:size(y,1)
       y(p,labels(p)) = 1;
   end
end

if strcmp(name,'faces')
   imageDim = 20;
   load('faces.mat');
   load('nonfaces.mat');
   images = [faces ; nonfaces];
   images = double(images)/255;
   X = reshape(images',imageDim,imageDim,1,[]);

   y = ones(size(faces,1),1);
   y = [y ; ones(size(nonfaces,1),1)*2];   
   
   %randpermutation
   index = randperm(size(X,4));
   X = X(:,:,:,index);
   yy = zeros(2,size(y,1));
   for p = 1:size(y,1)
       yy(y(p),p) = 1;
   end
   y= yy';
end

if strcmp(name,'face2')
    
    %addpath('./FaceData/');
   
   % [ X1,y1] = LoadFaces();
   % load errdata;
   % [ X2,y2] = LoadNoFaces();
    
   % index = randperm(size(X2,4));
   % X2 = X2(:,:,:,index);
   % y2 = y2(index,:);
    
   % en = size(X,4);
    
   % disp(en);
   % X2(:,:,1,1:en) = X;
   load picdata;
   load errdata; 
   load model2;
   index = labels(:,2) ==1;
   
   tmp = images(:,:,:,index);
   tmp = mergeFeatureMap(tmp,X);

   tres = zeros(size(tmp,4),2);
   %减少model无用数据，才可以并行访问
   model.log = 0;
   model.corind = 0;
   
   parfor i = 1 : size(tmp,4)
       pp = cnnCalcnet( model, tmp(:,:,:,i));
       tres(i,:) = pp{end};
   end
   
   [~,index] = sort(tres,1);
   index = flip(index(:,1));
   
   index = index(1:40100);
   
   disp([tres(index(1),1) min(min(tres(index,1)))]);
   
   X1 = tmp(:,:,:,index);  %非人脸样本
   X2 = images(:,:,:, labels(:,1)==1);  %人脸样本
    %X = zeros(size(X1)+[0 0 0 size(X2,4)]);
   %y = zeros(size(X,4),2);
   
   % X(:,:,1,1:size(X1,4)) = X1;
   % y(1:size(X1,4),:) = y1;
   % X(:,:,1,size(X1,4)+1:end) = X2;
   %  y(size(X1,4)+1:end,:) = y2;
   X = mergeFeatureMap(X1,X2);
   y = zeros(size(X,4),2);
   y(1:size(X1,4),2) = 1;
   y(size(X1,4)+1:end,1) = 1;
   % index = randperm(size(X,4));
    %X = X(:,:,:,index);
    %y = y(index,:);
    
   fprintf('Loaded %d.\n',size(X,4));
end

end

