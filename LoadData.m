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
   
   X = zeros(320,240,1,N);
   
    
    
end

end

