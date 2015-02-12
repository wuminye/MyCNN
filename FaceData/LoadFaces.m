function [ X,yx,yy] = LoadFaces()
%LOADFACES Summary of this function goes here
%   Detailed explanation goes here
  load('Database');
  dir = './FaceData/pic/';
  cy = 16;
  cx = 12;
  step = 5;
  scale = 6;
  rx = 120;
  ry = 160;
  dx = rx / 12;
  dy = ry / 16;
  N = Database.cnt ;
  %N = 2 ;
  
  X = zeros(rx,ry,1,0);
  yy = zeros(0,cy+1);
  yx = zeros(0,cx+1);
  fprintf('begin read IMG...\n');
  for i = 1 : N
      if rand >0.29
          continue;
      end
      fprintf('\r%5d\r',i);
      data = Database.data{i};
      [px,py] = getpoint(data.data{1});
      px = px./(scale);
      py = py./(scale);
      F = imread([dir data.filename]);
      F = double(imresize(F,1/scale))/255;
      [sx sy] = size(F);
      for p = 1:step:rx-sx
          for q = 1:step:ry-sy
            if rand >0.9
             X(:,:,1,end+1) = rand(rx,ry,1);
             X(p:p+sx-1,q:q+sy-1,1,end) = F;
          %   subplot(2,2,1);
           %  imshow(X(:,:,1,end));
           %  hold on;
          %   scatter(px+q-1,py+p-1, 'g.');
             yx(end+1,ceil((py+p-1)/dy)+1) = 1;
             yy(end+1,ceil((px+q-1)/dx)+1) = 1;
           %  subplot(2,2,2);
            % imagesc(yx(end,:)',[0 1]);
            % subplot(2,2,3);
            % imagesc(yy(end,:),[0 1]);
            
            %·´Ñù±¾
             if rand<0.1
                 X(:,:,1,end+1) = rand(rx,ry,1);
                 X(p:p+sx-1,q:q+sy-1,1,end) = F(1);
                 yx(end+1,1) = 1;
                 yy(end+1,1) = 1;
             end
            end
          end
      end
      
  end
  
  
end


function [px,py] = getpoint(data)

t = data.face{1}.position;
px = data.img_width*t.center.x/100;
py = t.center.y*data.img_height/100;

end
