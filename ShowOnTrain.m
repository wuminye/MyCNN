load model2;
[images , labels] = LoadData(model.dataname);
N = size(images,4);
while true
    %load model2;
    for i = 1 : 10
      k = ceil(N*rand);
     ShowLayer( model, images(:,:,1,k) ,labels(k,:) );
     pause(5);
    end
end