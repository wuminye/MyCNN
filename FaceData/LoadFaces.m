function [ X,y] = LoadFaces()
% 加载人脸样本
%   Detailed explanation goes here
  load('Database');
  %dir = './FaceData/pic/';
 dir = './pic/';
 
  rx = 36; %目标高度
  ry = 32; %目标宽度
  
  N = Database.cnt ;
  N = 100 ;
  
  X = zeros(rx,ry,1,0);
  y = zeros(0,2);
 
  fprintf('begin read IMG...\n');
  for i = 1 : N
      data = Database.data{i};
      ang = str2double(data.filename(6:8));
      %大角度样本丢弃
      if abs(ang)>=40
          continue;
      end
      fprintf('\r%5d\r',i);
      
      [px,py,h,w] = getpoint(data.data{1});
       w = w*1.15;
      if abs(ang)>=25
           w = w *1.1;
      end
 
      h = w*rx/ry;

      F = imread([dir data.filename]);
      F = double(F)/255;

      nF = F(ceil(py-h/2:py+h/2),ceil(px - w/2:px + w/2));
      nF = imresize(nF,[rx,ry]);
      X(:,:,1,end+1) = nF;
      y(end+1,1) = 1;
      X(:,:,1,end+1) = medfilt2(nF);
      y(end+1,1) = 1;
      imshow(nF);
  end
  
  
end


function [px,py,h,w] = getpoint(data)

t = data.face{1}.position;
px = data.img_width*t.center.x/100;
py = t.center.y*data.img_height/100;
h = t.height*data.img_height/100;
w = t.width*data.img_width/100;
end
