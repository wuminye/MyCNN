function [ y ] = cnnANN( input , w , b )
%CNNANN Summary of this function goes here
%   Detailed explanation goes here

input = squeeze(input);
assert(size(input ,2) == 1, ['Dims of input error  ', '']);
assert(size(w ,1) == size(b,1), ['Dims of w and b error  ', '']);
assert(size(input ,1) == size(w ,2), ['Dims of input and w error  ', '']);


y = w * input + b;

y = ActiveFunction(y);
end

