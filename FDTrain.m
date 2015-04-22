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

    save picdata images labels;
end
