addpath('./MNIST/');
addpath('./Core/');
addpath('./Util/');
model = InitCNNModel();
save model model;


while true

    TrainModel;
    %{
    [ X,y ] = LoadErrNonFaces( model );
    if size(X,4)>60000
       X = X(:,:,:,1:60000);
       y = y(1:60000,:);
    end
    save errdata X y ;
   
    if size(X,4)<50
        break;
    end
    [images , labels] = PrepareData(40000,40000);
    if strcmp(model.type, 'small') ==1
         save picdatasmall images labels;
    else    
        save picdata images labels;
       % inputimg = imresize(data,scale);
    end
    model.lambda = model.lambda * 100;
    save model model;
    %}
end
