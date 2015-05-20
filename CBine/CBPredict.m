function [ res ] = CBPredict( models,data ,model )

if size(data,1)==1   %Œƒº˛ ‰»Î
    data = imread(data);
    data = rgb2gray(data);
    data = imresize(data,1000/max(size(data)));
    data = double(data)/255;
end   

ress = [];

for i = 1:length(models)
    [tem ,row] = CalcFeature( models{i},data );
    ress = [ress;tem];
end
ANN_w = model.sublayer{end}.subnet{end}.model.Layer{2}.w;
ANN_b = model.sublayer{end}.subnet{end}.model.Layer{2}.b;
pres =ActiveFunction(ANN_w*ress + repmat(ANN_b,1,size(ress,2)));   

SM_w = model.sublayer{end}.subnet{end}.model.Layer{end}.w;

res = exp( SM_w*pres);
res = res./repmat(sum(res),2,1);
b = reshape(res(1,:),row,[]);

imshow(b);

end


function [ pres ,row] = CalcFeature(model,data)
     [~ , featuremap] = cnnCalcForward( model ,data ,1 );
     model = model.sublayer{end}.subnet{end}.model;
     res = cell(length(model.Layer),1);
     res{1} = featuremap;
     
     endmark = 1;
     for i  = 2 : length(model.Layer)
       
         cur = model.Layer{i}.type;
         if strcmp(cur,'Reshape')
             window = model.Layer{i-1}.out;
             res{i} = zeros(size(res{i-1},1)-window(1)+1,size(res{i-1},2)-window(2)+1,model.Layer{i}.kernelsize(3));
             for x = 1:size(res{i-1},1)-window(1)+1
                 for y = 1 : size(res{i-1},2)-window(2)+1
                     tmp = res{i-1}(x:x+window(1)-1,y:y+window(2)-1,:);
                    
                     res{i}(x,y,:) = reshape(tmp,model.Layer{i}.kernelsize);
                 end
             end
         end
         if strcmp(cur,'SoftMax')
            res{i} = cnnSoftMax(res{i-1},model.Layer{i}.w);
         end
         if strcmp(cur,'Conv')
             res{i} = cnnConv(res{i-1},model.Layer{i}.w,model.Layer{i}.b,...
                              model.Layer{i}.connector);
         end
         if strcmp(cur,'Convs')
             res{i} = cnnConvs(res{i-1},model.Layer{i}.w,model.Layer{i}.b,...
                              model.Layer{i}.connector,model.Layer{i}.stride);
         end
          if strcmp(cur,'Pooling')
             res{i} = cnnPooling(res{i-1}, model.Layer{i}.kernel );
          end
         if strcmp(cur,'ANN')
             endmark = i-1;
             break;
         end
         
     end
     row = size(res{endmark},1);
     temp = zeros(size(res{endmark},3),size(res{endmark},1)*size(res{endmark},2));
     for p = 1 : size(res{endmark},3)
        temp(p,:) = reshape(res{endmark}(:,:,p),1,[]); 
     end
     i = endmark+1;
     pres =ActiveFunction(model.Layer{i}.w*temp + repmat(model.Layer{i}.b,1,size(temp,2)));   
end
    
