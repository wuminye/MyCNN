
%生成picdata 训练文件

addpath('./FaceData/');
addpath('./Core/');
addpath('./Util/');

pn = 100000;
nn = 100000;
fprintf('generating training data: %d face ; %d nonface\n',pn,nn);
[ X , y ] = LoadData(pn,nn);

save picdata X y

fprintf('Done.\n');

