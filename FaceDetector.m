function [ cp , faces ] = FaceDetector(model, data )

addpath('./FaceData/');
addpath('./Core/');
addpath('./Util/');

if size(data,1)==1   %文件输入
    data = imread(data);
    data = rgb2gray(data);
    data = imresize(data,800/max(size(data)));
    data = double(data)/255;
end   

cp = 0;

%load model2

[res,~,scales] = cnnPredict(model,data,1);

%先处理原始热度图
for i = 1 : length(res)

    %res{i}(res{i}<=0.1) = 0;
    %imshow(res{i});
    t= res{i}*0.9;
    if i~= 1
        t = t + 0.1*imresize(res{i-1},size(t));
    end
    if i~= length(res)
        t = t + 0.1*imresize(res{i+1},size(t));
    end
    %t = medfilt2(t,[2 2]);
    figure;
    subplot(2,1,1);
    imshow(t);
   subplot(2,1,2);
    [faces]=splitIMG(model,imresize(data,scales(i)),t,[0.4 0.6]);
end

%{
imgs = res{1};
for i = 2 : length(res)
    
    tem = myimTransform(res{i},size(imgs));
    imgs = imgs + tem; 
end
   
   

   t = imgs;
   
   %t = medfilt2(t,[2 2]);
   %t = imgs>0.3;
   %t = medfilt2(t,[2 2]);
   figure;
   imshow(t);
   
   
   [faces]=splitIMG(model,data,t,[0.4 0.70]);
 %}
end

function [X]=splitIMG(model,img,data,rate)


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
    for scale = 0.7:0.2:1.3
        %坐标转换到原图
        tcx = ceil(tx*(sx-36)/dx - rx*scale/2 + 16);
        tcy = ceil(ty*(sy-36)/dy - ry*scale/2 + 16);
        %cx = ceil(tx*(sx)/dx - rx*scale/2 );
        %cy = ceil(ty*(sy)/dy - ry*scale/2 );
        ccnt = 0; % 如果已检测到人脸，就不再周围区域搜索
       for cx = tcx:5:tcx
           
           if ccnt >2
                break;
            end
           
           for cy = tcy:3:tcy+3
               
           if ccnt >2
                break;
            end
           
                if cx<1 || cy<1 || ceil(cx+rx*scale-1)>sx || ceil(cy+ry*scale-1)>sy
                   continue;
                end
                timg = img(cx:ceil(cx+rx*scale-1),cy:ceil(cy+ry*scale-1));
                timg = imresize(timg,[rx,ry]);
                res = cnnCalcForward( model, timg);
                rr = res{end}{end}{end};
                imshow(timg);
              % disp([rr(1) rr(2)]);
                if rr(1)>rate(2)
                    
                   X(:,:,1,end+1) = timg;
                    rectangle('Position', ...
                    [cy, cx, ry*scale, rx*scale], ...
                    'Curvature', 0.4, 'LineWidth',1, 'EdgeColor', 'blue');
                   drawnow;
                   fprintf('%d %d %f %f \n',cx,cy,rr(1), rr(2));
                  % imshow(timg);
                  % pause;
                  ccnt = ccnt + 1;
                end
           end
       end
   end
    
end

end

function res = myimTransform(img,dsize)
   [m ,n] = size(img);
   scale = dsize./[m ,n];
   res = zeros(dsize);

   for x = 1 : m
       for y = 1: n
           dd = scale.*[x y];
           if (dd(1)>0.5 && dd(2)>0.5)
               dd = round(dd);
               res(dd(1),dd(2)) = img(x,y);
           end
           
       end
   end
end