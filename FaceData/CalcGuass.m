function [ Z , cx , cy ] = CalcGuass( data ,  scale )
%CALCGUASS Summary of this function goes here
%   Detailed explanation goes here
imsize = [data.img_width  data.img_height];
Z = zeros(flip(imsize/scale));
x = 1:size(Z,2);
y = 1:size(Z,1);
[xx,yy] = meshgrid(x,y);

for i = 1 : length(data.face)
   t = data.face{i}.position;
   
   th = (t.height*data.img_height+t.width*data.img_width)/(800*scale);
   cy = data.img_width*t.center.x/scale/100;
   cx = t.center.y*data.img_height/scale/100;
   zz = exp(-( (xx-cy).^2 + (yy-cx).^2 )/(2*th.^2));
   zz(zz<0.5) = 0;
 %  zz=zz*1.4;
   zz(zz>0.6)=1;
   Z = Z + zz;
end
   [cx] = floor([cx ]);
   cy = floor(cy);
end

