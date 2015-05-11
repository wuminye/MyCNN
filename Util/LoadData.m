function [ X , y ] = LoadData(pn,nn )


   addpath('./FaceData/');

   [ X1,y1] = LoadFaces();
   [ X2,y2] = LoadNoFaces();

   load morphdata;
   %随机选取一些morph样本
    index = randperm(size(images,4),14000);
    images = images(:,:,:,index);
   X1 = mergeFeatureMap(X1,images);

   load fddb;
   X1 = mergeFeatureMap(X1,X);

    index = randperm(size(X1,4));
    X1 = X1(:,:,:,index);



    X1 =  X1(:,:,:,1:pn);

    index = randperm(size(X2,4));
    X2 = X2(:,:,:,index);


    X2 =  X2(:,:,:,1:nn);

    X = mergeFeatureMap(X1,X2);

    y = zeros(size(X,4),2);
    y(1:size(X1,4),1) = 1;
    y(size(X1,4)+1:end,2) = 1;



end
