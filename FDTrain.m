addpath('./FaceData/');
addpath('./Core/');
addpath('./Util/');
model = InitCNNModel();
save model model;


while true

    TrainModel;
    [ X,y ] = LoadErrNonFaces( model );
    save errdata X y ;
    if size(X,4)<50
        break;
    end
    [images , labels] = PrepareData(100000,100000);
    if strcmp(model.type, 'small') ==1
         save picdatasmall images labels;
    else    
        save picdata images labels;
       % inputimg = imresize(data,scale);
    end
    
end
