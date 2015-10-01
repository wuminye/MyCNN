function [ res ] = cnnCalcReg( model )
%CNNCALCREG Summary of this function goes here
%   Detailed explanation goes here
num = length(model.Layer);
res = 0;
for i = 2 : num
    t = model.Layer{i};
    cur = t.type;
    if strcmp(cur,'Conv') 
        tem = t.w.^2;
        res = res + sum(tem(:));
        
         % tem = t.beta.^2;
       % res = res + sum(tem(:));
        
        for q = 1 : size(model.Layer{i}.w,4)
          for p = 1 : size(model.Layer{i}.w,3)
              ahpla = exp(model.Layer{i}.beta(p,q))/ sum(exp(model.Layer{i}.beta(:,q)));
              res = res + abs(ahpla);
          end
        end
        
    end
    
    if strcmp(cur,'Convs')
        tem = t.w.^2;
        res = res + sum(tem(:));
    end
    
    if strcmp(cur,'ANN') 
        
    end
    
    if strcmp(cur,'SoftMax')
        %tem = t.w.^2;
       tem = abs(t.w);
        res = res + sum(tem(:));
    end
    
    
     if strcmp(cur,'Pooling') 
        tem = t.w.^2;
        res = res + sum(tem(:));
    end
    
    
end

end

