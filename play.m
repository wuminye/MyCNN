addpath('./MNIST/');
addpath('./Core/');
addpath('./Util/');

load model
load testdata

tt=model.sublayer{2}.subnet{1}.model;
while true
    n = ceil(rand *10000);
    ShowLayer(tt,images(:,:,:,n),labels(n,:));
    pause(0.8);
    
end 