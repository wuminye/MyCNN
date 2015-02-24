function [ y ] = ActiveFunction( x )
%ACTIVEFUNCTION Summary of this function goes here
%   Detailed explanation goes here

%y = 1.0 ./ (1.0 + exp(-x));
 y = max(0,x);
end

