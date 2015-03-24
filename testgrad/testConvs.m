function testConvs()

a = rand(4,4);
stride = 2;
w = rand(3,3);

[feature , cost] = pf(a,w,stride);
dw =  bp(feature,a,w,stride);
eps = 1e-6;
k= zeros(9,1);
for i = 1:9
    w(i) = w(i)+eps;
    [feature , cost1] = pf(a,w,stride);
    w(i) = w(i)-eps*2;
    [feature , cost2] = pf(a,w,stride);
    k(i) = (cost1-cost2)/(2*eps);
end
   disp(k');
   disp(dw(:)');
end
function [x,x2] = pf(a,w,stride)
   addpath('../');
   tem = conv2(a,rot90(w,2),'same');
   tem = tem(1:stride:end,1:stride:end);
   x = ActiveFunction(tem);
   x2= sum(x(:));
end
function [dw]  = bp(feature,a,w,stride)

t= deActiveFunction(feature);
[x1, y1] = size( a);

tem = zeros(size(a));
tem(1:stride:end,1:stride:end)...
             = t;

hk = floor(size(w)/2);
nf = zeros(size(tem,1)+hk(1)*2,...
            size(tem,2)+hk(2)*2);
nf(hk(1)+1:hk(1)+x1,hk(2)+1:hk(2)+y1) = a;
dw = conv2(nf, rot90(rot90(tem)),'vaild');

end