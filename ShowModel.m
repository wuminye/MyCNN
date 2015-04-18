function [ model ] = ShowModel( model )

fprintf('========================');
fprintf('lambda:%.6f\n',model.lambda);
fprintf('number of data for training:%d\n',model.num_train);
fprintf('------------------------');
t= model.Layer;
for i = 1: length(t)
   fprintf('Layer %d\n',i);
   disp(t{i});
   figure;
   if strcmp(t{i}.type,'Conv')
       subplot(sx,sy,cnt);
   end
    
end


end

