
%����picdata ѵ���ļ�

addpath('./FaceData/');
addpath('./Core/');
addpath('./Util/');

pn = 100000;
nn = 100000;
fprintf('generating training data: %d face ; %d nonface\n',pn,nn);
[ X , y ] = LoadData(pn,nn);

images = X;
labels = y;

save picdata images labels

fprintf('Done.\n');

