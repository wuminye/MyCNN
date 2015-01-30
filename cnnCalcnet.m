function [ res ] = cnnCalcnet( model ,data )
%CNNCALCNET Summary of this function goes here
%   Detailed explanation goes here
 res = cell(length(model.Layer),1);
 res{1} = data;
 for i  = 2 : length(model.Layer)
     cur = model.Layer{i}.type;
     if strcmp(cur,'Reshape')
         res{i} = cnnReshape(res{i-1},model.Layer{i}.kernelsize); 
     end
     
     if strcmp(cur,'SoftMax')
        res{i} = cnnSoftMax(res{i-1},model.Layer{i}.w);
     end
     
     if strcmp(cur,'Conv')
         res{i} = cnnConv(res{i-1},model.Layer{i}.w,model.Layer{i}.b,...
                          model.Layer{i}.connector);
     end
     
      if strcmp(cur,'Pooling')
         res{i} = cnnPooling(res{i-1}, model.Layer{i}.kernel );
      end
     
     if strcmp(cur,'ANN')
         res{i} = cnnANN(res{i-1}, model.Layer{i}.w, model.Layer{i}.b );
     end
 end

end

