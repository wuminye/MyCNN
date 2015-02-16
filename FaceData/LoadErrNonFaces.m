function [ X,y ] = LoadErrNonFaces( model )
%LOADERRNONFACES Summary of this function goes here
%   Detailed explanation goes here

dir1 = './FaceData/nofaces/';
%dir1 = './nofaces/';

Files = dir(fullfile(dir1,'*'));
Files = Files(3:end);
LengthFiles = length(Files);

rx = 36; %目标高度
ry = 32; %目标宽度

X = zeros(rx,ry,1,0);
fprintf('Begin Calc NonFaces Errors ...\n');
for i = 1 : LengthFiles
   fprintf('\r%5d\r',i);
   [ res ,tX ] = cnnPredict( model,strcat(dir1,Files(i).name));
    temp = X;
    X = zeros(size(tX,1),size(tX,2),size(tX,3),size(tX,4)+size(temp,4));
    X(:,:,1,1:size(temp,4)) = temp;
    X(:,:,1,size(temp,4)+1:end) = tX;
   
    
end

y = zeros(size(X,4),2);
y(:,2) = 1;
fprintf('Calc %d Err-nonfaces.\n',size(X,4));
end

