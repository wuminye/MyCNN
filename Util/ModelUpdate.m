function [ model ] = ModelUpdate( model )
   tem  = InitCNNModel();
   tem.log = model.log;
   tem.logn = model.logn;
   tem.lambda = model.lambda;
   model.log = [];
   model.logn = 0;
   model.corind = [];
   tem.sublayer = cell(2,1);
   tem.sublayer{1}.subnet{1} = 0;
   tem.sublayer{2}.subnet{1}.model = model;
   tem.sublayer{2}.connect = ones(1,length(tem.sublayer{2}.subnet));
   
   model = tem;

end

