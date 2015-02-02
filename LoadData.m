function [ X, y ] = LoadData( name )
%LOADDATA Summary of this function goes here
%   Detailed explanation goes here
if strcmp(name,'MNIST')
   imageDim = 28;
   images = loadMNISTImages('train-images.idx3-ubyte');
   images = reshape(images,imageDim,imageDim,1,[]);
   X = images;
   disp(size(images));
   labels = loadMNISTLabels('train-labels.idx1-ubyte');
   labels(labels==0) =10;
   y =  labels;
end

end

