function [ res ] = cnnCalcSubnet( model ,data,OnTrain )
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
     
     if strcmp(cur,'Convs')
        res{i} = cnnConvs(res{i-1},model.Layer{i}.w,model.Layer{i}.b,...
                          model.Layer{i}.connector, model.Layer{i}.stride);
     end
     
     if strcmp(cur,'Conv')
         res{i} = cnnConv(res{i-1},model.Layer{i}.w,model.Layer{i}.b,...
                          model.Layer{i}.connector,model.Layer{i}.beta);
     end
     
      if strcmp(cur,'Pooling')
         res{i} = cnnPooling(res{i-1}, model.Layer{i}.kernel ,model.Layer{i}.w, model.Layer{i}.b);
      end
     
     if strcmp(cur,'ANN')
         if  model.Layer{i}.dropout.enable == 1 && OnTrain == 1
             res{i-1} = res{i-1} / model.Layer{i}.dropout.p;
         end
         res{i} = cnnANN(res{i-1}, model.Layer{i}.w, model.Layer{i}.b,model.Layer{i}.mask );
     end
 end

end

