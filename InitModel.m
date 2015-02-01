function [ model ] = InitModel( Layer )
%INITMODEL Summary of this function goes here
%   Detailed explanation goes here

num_layer = length(Layer);

for i = 1 : num_layer
   cur = Layer{i}.type;
   %{
   if strcmp(cur,'Input')
       Layer{i}.out = size(input);
   end
   %}
   if strcmp(cur,'Reshape')
        Layer{i}.out =  Layer{i}.kernelsize;
   end
   
   if strcmp(cur,'SoftMax')
      r = sqrt(6)/(max(Layer{i-1}.out(:)) + Layer{i}.out(1));
      p = max(Layer{i-1}.out(:));
      Layer{i}.w = rand(Layer{i}.out(1),p)*2*r - r ;
   end
   
   if strcmp(cur,'Conv')
       Layer{i}.out = [Layer{i-1}.out(1)- Layer{i}.kernelsize(1)+1,...
                       Layer{i-1}.out(2)- Layer{i}.kernelsize(2)+1, Layer{i}.mapnum];
       r = sqrt(6)/(Layer{i}.kernelsize*Layer{i}.kernelsize' + 1);
       %r = sqrt(6)/(Layer{i-1}.out(1)*Layer{i-1}.out(2) + 1);
       %featuremap多个卷积核 --- 4D
       Layer{i}.w = rand([ Layer{i}.kernelsize ,Layer{i-1}.out(3) ,Layer{i}.mapnum ])*2*r - r;
       Layer{i}.b = rand(Layer{i}.mapnum,1)*2*r - r ;   %实数偏置
       Layer{i}.connector = ones(Layer{i}.mapnum,Layer{i-1}.out(3));
   end
    
   if strcmp(cur,'Pooling')
       Layer{i}.out = floor([Layer{i-1}.out(1:2)./Layer{i}.kernelsize , Layer{i-1}.out(3)]);
       Layer{i}.kernel.x = Layer{i}.kernelsize(1);
       Layer{i}.kernel.y = Layer{i}.kernelsize(2);
   end
   
   if strcmp(cur,'ANN')
      pp = max(Layer{i-1}.out(:));
      p = max(Layer{i}.out(:));
      r = sqrt(6)/(pp + p);
      Layer{i}.w = rand(p,pp)*2*r - r ;
      Layer{i}.b = rand(p,1)*2*r - r;
   end
end

model.Layer = Layer;

end

