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
   y =  labels;
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
   y = y(index);
end

end

