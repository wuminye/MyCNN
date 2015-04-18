function [ y ] = cnnSoftMax( input , w )

assert(size(input ,1) == 1 && size(input ,2) == 1, ['Dims of input error  ', '']);
assert(size(input ,3) == size(w ,2), ['Dims of input and w error  ', '']);

tem =  input(1,1,:);
tem =  tem(:);

y = exp(w*tem);
y = y./sum(y);

% featuremap ±ê×¼»¯
y = reshape(y,1,1,[]);
end

