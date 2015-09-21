function [ y ] = cnnANN( input , w, b ,mask)
%CNNANN Summary of this function goes here
%   Detailed explanation goes here

%input = squeeze(input); 保持规范化
assert(size(input ,1) == 1, ['Dims of input error  ', '']);
assert(size(input ,2) == 1, ['Dims of input error  ', '']);
assert(size(w ,1) == size(b,1), ['Dims of w and b error  ', '']);
assert(size(input ,3) == size(w ,2), ['Dims of input and w error  ', '']);

tem =  input(1,1,:);
y = (mask.*w) * tem(:) + b;

y = ActiveFunction(y);
y = reshape(y,1,1,[]);
end

