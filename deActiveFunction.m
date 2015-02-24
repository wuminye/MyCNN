function [ res ] = deActiveFunction( a )
%DEACTIVEFUNCTION Summary of this function goes here
%   Detailed explanation goes here
   %res = a.*(1-a);
   res = double(a>0);
end

