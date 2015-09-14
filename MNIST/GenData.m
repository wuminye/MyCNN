images = reshape(loadMNISTImages('train-images.idx3-ubyte'),28,28,1,60000);
tl = loadMNISTLabels('train-labels.idx1-ubyte');

labels = zeros(size(tl,1),10);
for i = 1:size(tl,1)
   labels(i,tl(i)+1) = 1; 
end

save picdata.mat images labels

images = reshape(loadMNISTImages('t10k-images.idx3-ubyte'),28,28,1,10000);
tl = loadMNISTLabels('t10k-labels.idx1-ubyte');
labels = zeros(size(tl,1),10);
for i = 1:size(tl,1)
   labels(i,tl(i)+1) = 1; 
end

save testdata.mat images labels
