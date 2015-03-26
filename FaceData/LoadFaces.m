function [ X,y] = LoadFaces()
% ������������
%   Detailed explanation goes here
  load('Database');
 dir = './FaceData/pic/';
 %dir = './pic/';
 dir2 = './FaceData/lfw/';
 %dir2 = './lfw/';


  rx = 36; %Ŀ���߶�
  ry = 32; %Ŀ������

  N = Database.cnt ;
  %N = 100 ;

  X = zeros(rx,ry,1,0);
  y = zeros(0,2);

  fprintf('begin read IMG...\n');
  for i = 1 : N
      data = Database.data{i};
      ang = str2double(data.filename(6:8));
      %���Ƕ���������
      if abs(ang)>=30
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
      X(:,:,1,end+1) = medfilt2(nF,[2 2]);
      y(end+1,1) = 1;
      %imshow(nF);
  end

  fprintf('begin read LFW...\n');
  load('Database2');

  N = Database.cnt ;
  for i = 1 : N
      data = Database.data{i};

      fprintf('\r%5d\r',i);

      [px,py,h,w] = getpoint(data.data{1});



      w = w*1.15;

      h = w*rx/ry;

      F = rgb2gray(imread([dir2 data.filename]));
      F = double(F)/255;

      %��ֹԽ��
      if py-h/2<1 || py+h/2>size(F,1) || px - w/2<1 || px + w/2>size(F,2)
          continue;
      end

      nF = F(ceil(py-h/2:py+h/2),ceil(px - w/2:px + w/2));
      nF = imresize(nF,[rx,ry]);
      X(:,:,1,end+1) = nF;
      y(end+1,1) = 1;
      X(:,:,1,end+1) = medfilt2(nF,[2 2]);
      y(end+1,1) = 1;
     % imshow(nF);
  end

  fprintf('Loaded %d faces.\n',size(X,4));

end


function [px,py,h,w] = getpoint(data)

t = data.face{1}.position;
px = data.img_width*t.center.x/100;
py = t.center.y*data.img_height/100;
h = t.height*data.img_height/100;
w = t.width*data.img_width/100;
end
