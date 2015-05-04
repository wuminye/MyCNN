[ models ] = Loadmodels(5);
fprintf('models are loaded.\n');
load picdata;
fprintf('picdata are loaded.\n');
[ X ] = PrepareFeature( models, images);
fprintf('Feature is Prepared.\n');
y = labels;
save CBdata X y;
fprintf('Data is saved.\n');
model = ModelInit( 25*5 );

save CBmodel model;
fprintf('CBmodel is saved.\n');