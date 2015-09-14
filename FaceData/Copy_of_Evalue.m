filedir = '.\FaceData\lfw\';
Files = dir(fullfile(filedir,'*'));
Files = Files(3:end);
LengthFiles = length(Files);
fprintf('<< %d  files totally>>\n',LengthFiles);

load vfmodel.mat
load bgmodel.mat

recd = cell(LengthFiles,1);

for i = 1:LengthFiles
   filename = strcat(filedir,Files(i).name);
    data = imread(filename);
    data = rgb2gray(data);
    data = imresize(data,36/max(size(data)));
    data = double(data)/255;
    
    res1 = cnnCalcForward(model,data);
    res2 = cnnCalcForward(tt,data);
    if res1{end}{end}{end}(1)<0.8 || res2{end}{end}{end}(1)<0.8
        disp(i);
        disp(Files(i).name) ;
       disp([res1{end}{end}{end}(1) res2{end}{end}{end}(1)]);
    end
    output1 = res1{end}{end}{end-1}(:);
    output2 = res2{end}{end}{end-1}(:);
    
    recd{i}.filename = Files(i).name;
    recd{i}.feature1 =  output1;
    recd{i}.feature2 =  output2;
    recd{i}.confid = [res1{end}{end}{end}(1) res2{end}{end}{end}(1)];
end
