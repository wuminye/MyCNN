[ models ] = Loadmodels(4);
fprintf('models are loaded.\n');
load picdata;
fprintf('picdata are loaded.\n');
[ X ,CNT ] = PrepareFeature( models, images);
fprintf('Feature is Prepared.  %d \n',CNT);
y = labels;
save CBdata X y;
fprintf('Data is saved.\n');
model = ModelInit( CNT );

save CBmodel model;
fprintf('CBmodel is saved.\n');