addpath('./FaceData/');
model = GetModel('face2');
save model2 model;


while true

    main2;
    [ X,y ] = LoadErrNonFaces( model );
    save errdata X y ;
    if size(X,4)<50
        break;
    end
    [images , labels] = LoadData(model.dataname);
    %images = images(:,:,:,1:130000);
    %labels = labels(1:130000,:);
    save picdata images labels;
end