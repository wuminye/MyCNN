function [ X,y ] = LoadNoFaces( )
%LOADNOFACES Summary of this function goes here
%   Detailed explanation goes here

  %dir = './FaceData/nofaces/';
 dir1 = './nofaces/';

Files = dir(fullfile(dir1,'*'));
Files = Files(3:end);
LengthFiles = length(Files);

rx = 36; %目标高度
ry = 32; %目标宽度
step = 50;

X = zeros(rx,ry,1,0);
y = zeros(0,2);

fprintf('begin read IMG...\n');
for i = 1 : LengthFiles
    F = rgb2gray(imread(strcat(dir1,Files(i).name)));
    F = double(F)/255;
    F = imresize(F,1600/max(size(F)));
    scale = 1;
    for j = 1 : 3
       nF = imresize(F,scale);
       [sx, sy]=size(nF);
       for ix = 1:step:sx-rx+1
           for iy = 1:step:sy-ry+1
               
               X(:,:,1,end+1)=nF(ix:ix+rx-1,iy:iy+ry-1);
               y(end+1,2) = 1;
               imshow(X(:,:,1,end));
               X(:,:,1,end+1)=medfilt2(nF(ix:ix+rx-1,iy:iy+ry-1));
               y(end+1,2) = 1;
               imshow(X(:,:,1,end));
           end
       end
       scale = scale*0.6;
    end
    
end

end

