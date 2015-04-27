function [ X, y ] = PrepareData(pn,nn)



   addpath('./FaceData/');

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
   load model;
   
   if strcmp(model.type, 'small') ==1
        X =NormalizeIMG( X ,0.5);
   end
   index = labels(:,2) ==1;

   tmp = images(:,:,:,index);
   tmp = mergeFeatureMap(tmp,X);

   tres = zeros(size(tmp,4),2);
   %减少model无用数据，才可以并行访问
   model.log = 0;
   model.corind = 0;

   parfor i = 1 : size(tmp,4)
       pp = cnnCalcForward( model, tmp(:,:,:,i));
       tres(i,:) = pp{end}{end}{end}(:)';
   end

   %============================
   %构造非人脸样本
   [~,index] = sort(tres,1);
   index = flip(index(:,1));
   index = index(1:nn);

   disp([tres(index(1),1) min(min(tres(index,1)))]);

   X1 = tmp(:,:,:,index);  %非人脸样本
   X2 = images(:,:,:, labels(:,1)==1);  %人脸样本

   index = randperm(size(X2,4),pn);
   X2 = X2(:,:,:,index);


   X = mergeFeatureMap(X1,X2);
   y = zeros(size(X,4),2);
   y(1:size(X1,4),2) = 1;
   y(size(X1,4)+1:end,1) = 1;


   fprintf('Loaded %d.\n',size(X,4));
end
