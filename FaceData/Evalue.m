filedir = '.\FaceData\lfw\';
Files = dir(fullfile(filedir,'*'));
Files = Files(3:end);
LengthFiles = length(Files);
fprintf('<< %d  files totally>>\n',LengthFiles);

load vfmodel.mat
load bgmodel.mat


for i = 1:LengthFiles
   filename = strcat(filedir,Files(i).name);
   [~,~,list]=FaceDetector(tt, filename ,model);
   drawnow;
   disp(list);
   fprintf('=========================\n');
    pause(0.5); 
    
end
