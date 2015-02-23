function [ cp , faces ] = FaceDetector(model, data )

if size(data,1)==1   %文件输入
    data = imread(data);
    data = rgb2gray(data);
    data = imresize(data,800/max(size(data)));
    data = double(data)/255;
end   

cp = 0;

%load model2

[res] = cnnPredict(model,data,1);

%先处理原始热度图
for i = 1 : length(res)

  %  res{i}(res{i}<=0.2) = 0;
    %imshow(res{i});
    %[faces]=splitIMG(model,data,res{i},[0.4 0.5]);
end


imgs = res{1};
for i = 2 : length(res)
    
    tem = imresize(res{i},size(imgs));
    imgs = imgs + tem; 
end
   
   

   t = imgs;
   t = imgs>0.2;
   t = medfilt2(t);
   figure;
   imshow(t);
   
   
   [faces]=splitIMG(model,data,t,[0.4 0.7]);
 
end

function [X]=splitIMG(model,img,data,rate)

figure;
imshow(img);
hold on;

[sx ,sy] = size(img); %原始图片大小
[dx , dy] = size(data); %热度图大小
[xx ,yy] = meshgrid([1:dx],[1:dy]);
xx = xx';
yy = yy';

rx = 36;
ry = 32;

X = zeros(rx,ry,1,0);

th = rate(1);

%获得热点坐标列表
x = xx(data>=th);
y = yy(data>=th);

N = size(x,1);


for i = 1:floor(N)

    tx = x(i);
    ty = y(i);
    for scale = 1:0.7:4
        %坐标转换到原图
        cx = ceil(tx*(sx-36)/dx - rx*scale/2 + 16);
        cy = ceil(ty*(sy-32)/dy - ry*scale/2 + 16);
        %cx = ceil(tx*(sx)/dx - rx*scale/2 );
        %cy = ceil(ty*(sy)/dy - ry*scale/2 );
    
       if cx<1 || cy<1 || ceil(cx+rx*scale-1)>sx || ceil(cy+ry*scale-1)>sy
           continue;
        end
        timg = img(cx:ceil(cx+rx*scale-1),cy:ceil(cy+ry*scale-1));
        timg = imresize(timg,[rx,ry]);
        res = cnnCalcnet( model, timg);
        rr = res{end};
       % imshow(img(cx:cx+rx-1,cy:cy+ry-1));
        if rr(1)>rate(2)

           X(:,:,1,end+1) = timg;
            rectangle('Position', ...
            [cy, cx, ry*scale, rx*scale], ...
            'Curvature', 0.4, 'LineWidth',1, 'EdgeColor', 'blue');
        drawnow;
          % imshow(timg);
          % pause;
        end
   end
    
end

end
