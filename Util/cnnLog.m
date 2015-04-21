function [ model ] = cnnLog(model,varargin)
%非线程安全

model.logn = model.logn + 1;
model.log{model.logn} = varargin ;

if varargin{1}(1)~='%'
   fprintf(varargin{1},varargin{2:end});
end
end

