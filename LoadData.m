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
   load('./FaceData/Database');
   addpath('./FaceData/');
   dir = './FaceData/pic/';
   N = Database.cnt ;
   
   %N = 50;
   
   X = zeros(120,160,1,N);
   y = zeros(N,16);
   fprintf('begin read IMG...\n');
   for i = 1 : N
    fprintf('\r%5d\r',i);
    data = Database.data{i};
    F = imread([dir data.filename]);
    X(:,:,1,i) = double(imresize(F,0.25))/255;
    [ta,tb,tc] = CalcGuass(data.data{1}, 40);
    y(i,:) = zeros(1,size(ta,2));
    y(i,tc)  = 1;
   end
end

end

