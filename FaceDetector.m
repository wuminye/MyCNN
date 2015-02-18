function [ cp , faces ] = FaceDetector( data )

if size(data,1)==1   %ÎÄ¼þÊäÈë
    data = imread(data);
    data = rgb2gray(data);
    data = imresize(data,2000/max(size(data)));
    data = double(data)/255;
end   


load model2

[res] = cnnPredict(model,data,1);

imgs = res{1};
for i = 2 : length(res)
    
    tem = imresize(res{i},size(imgs));
    imgs = imgs + tem; 
end

   imshow(imgs);
   t = imgs>0.4;
   t = medfilt2(t,[4 4]);
   figure;
   imshow(t);
   figure;
   
   [X]=splitIMG(model,data,t,[0.4 0.5]);
   
end

function [X]=splitIMG(model,img,data,rate)

[sx ,sy] = size(img);
[dx , dy] = size(data);
[xx ,yy] = meshgrid([1:dx],[1:dy]);
xx = xx';
yy = yy';

rx = 36;
ry = 32;

X = zeros(rx,ry,1,0);

th = rate(1);

x = xx(data>=th);
y = yy(data>=th);

N = size(x,1);


for i = 1:floor(N)
    tx = x(i);
    ty = y(i);
    
    cx = ceil(tx*sx/dx - rx/2);
    cy = ceil(ty*sy/dy - ry/2);
    
    if cx<1 || cy<1 || cx+rx-1>sx || cy+ry-1>sy
        continue;
    end

    res = cnnCalcnet( model, img(cx:cx+rx-1,cy:cy+ry-1));
    rr = res{end};
    imshow(img(cx:cx+rx-1,cy:cy+ry-1));
    if rr(1)>rate(2)

       X(:,:,1,end+1) = img(cx:cx+rx-1,cy:cy+ry-1);

    end
    
end

end
