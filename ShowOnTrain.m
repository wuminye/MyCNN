load model2;
load picdata2;

N = size(images,4);
while true
    load model2;
    fprintf('Loaded.\n');
    for i = 1 : 10
      k = ceil(N*rand);
     ShowLayer( model, images(:,:,1,k) ,labels(k,:) );
     drawnow;
     pause(5);
    end
end