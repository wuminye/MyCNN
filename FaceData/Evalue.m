
function Evalue()
load vfmodel.mat
load bgmodel.mat



dir1 = '../FDDB-folds/';
dir2 = '../';
Files = dir(fullfile(dir1,'*'));
Files = Files(3:end);
LengthFiles = length(Files);

cnt = 0;
X = zeros(36,32,1,0);
for i = 1 : LengthFiles
    if (strcmp(Files(i).name(14),'e')~=1)
        continue;
    end;
    fout = fopen([Files(i).name,'-res.txt'], 'w');
    fid = fopen(strcat(dir1,Files(i).name),'r');
    
    while ~feof(fid)
        tline = fgetl(fid);
          imname = tline;
        fprintf(fout,'%s\n',imname);
         fprintf('%s\n',imname);
        imgfn = strcat(strcat(dir2,tline),'.jpg');
        
        
        
        tline = fgetl(fid);
        [n] =  strread(tline,'%d');
        while n>0
            n = n-1;
             tline = fgetl(fid);
            [ major_axis_radius, minor_axis_radius, angle, center_x, center_y , t] =...
                         strread(tline,'%f %f %f %f %f %f');
        end
        
        [~,~,list]=FaceDetector(tt, imgfn ,model);
        fprintf(fout,'%d\n',size(list,1));
         fprintf('%d\n',size(list,1));
        [score,list] = GetScore(imgfn,list,model,tt);
        drawnow;
        for p = 1:size(list,1)
            fprintf(fout,'%f %f %f %f %f\n',list(p,2)-list(p,4)*1.1/2,list(p,1)-list(p,3)*1.25/2,list(p,4)*1.1,list(p,3)*1.25,score(p));
            fprintf('%f %f %f %f %f\n',list(p,2)-list(p,4)*1.1/2,list(p,1)-list(p,3)*1.25/2,list(p,4)*1.1,list(p,3)*1.25,score(p));
        end
       % pause(0.5); 
    end
    fclose(fout);
    

end


end
function [score,list] = GetScore(img,list,model,tt)
if (size(list,1)==0)
    score = [];
    return;
end
if size(img,1)==1   %ÎÄ¼şÊäÈë
    img = imread(img);
    if size(img,3) > 1
       img = rgb2gray(img);
    end
    img = double(img)/255;
end   
   list(:,1) = list(:,1) * size(img,1);
   list(:,2) = list(:,2) * size(img,2);
   list(:,3) = list(:,3) * size(img,1);
   list(:,4) = list(:,4) * size(img,2);
   score = zeros(0,1);
   for i = 1: size(list,1)
        cx = list(i,1) - list(i,3)/2;
        cy = list(i,2) - list(i,4)/2;
        if cx+list(i,3) > size(img,1) || cy+list(i,4)>size(img,2)
            score(end+1,1)=1;
            continue;
        end
        X = img(ceil(cx:cx+list(i,3)),ceil(cy:cy+list(i,4)));
        X = imresize(X,[36 32]);
        res = cnnCalcForward(model,X);
        output1 = res{end}{end}{end}(:);
        res = cnnCalcForward(tt,X);
        output2 = res{end}{end}{end}(:);
        
        score(end+1,1) = max([output1(1) output2(1)]);
        
   end
   score(score>1) = 1;
end
