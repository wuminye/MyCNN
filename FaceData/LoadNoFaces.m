function [ X,y ] = LoadNoFaces( )
%LOADNOFACES Summary of this function goes here
%   Detailed explanation goes here

dir1 = './FaceData/nofaces/';
%dir1 = './nofaces/';

Files = dir(fullfile(dir1,'*'));
Files = Files(3:end);
LengthFiles = length(Files);

rx = 36; %目标高度
ry = 32; %目标宽度
step = 80;

X = zeros(rx,ry,1,0);
y = zeros(0,2);

fprintf('begin read NonFaces...\n');
for i = 1 : LengthFiles
    fprintf('\r%5d\r',i);
    F = rgb2gray(imread(strcat(dir1,Files(i).name)));
    F = double(F)/255;
    F = imresize(F,2000/max(size(F)));
    scale = 1;
    step1 = step;
    for j = 1 : 4
       nF = imresize(F,scale);
       [sx, sy]=size(nF);
       for ix = 1:step1:sx-rx+1
           for iy = 1:step1:sy-ry+1
               
               X(:,:,1,end+1)=histeq(nF(ix:ix+rx-1,iy:iy+ry-1));
               y(end+1,2) = 1;
               %imshow(X(:,:,1,end));
               X(:,:,1,end+1)=histeq(medfilt2(nF(ix:ix+rx-1,iy:iy+ry-1)));
               y(end+1,2) = 1;
              % imshow(X(:,:,1,end));
           end
       end
       scale = scale*0.63;
       step1 = ceil(step1*0.70);
    end
    
end

fprintf('Loaded %d nonfaces.\n',size(X,4));

end

