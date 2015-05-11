
%生成picdata 训练文件

addpath('./FaceData/');
addpath('./Core/');
addpath('./Util/');

pn = 40000;
nn = 40000;
fprintf('generating training data: %d face ; %d nonface\n',pn,nn);
[ X , y ] = LoadData(pn,nn);

images = X;
labels = y;

save picdata images labels

fprintf('Done.\n');

