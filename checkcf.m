function [J, grad] = checkcf(Theta, X, y, lambda,Layer)
%COSTFUNCTION Summary of this function goes here
tTheta = cell(size(Layer,1)-1,1);
pos = 1;
top = size(Layer,1);
for i = 2:size(Layer,1)
    tTheta{i-1} = reshape(Theta(pos:pos+(Layer(i)*(Layer(i-1)+1))-1),Layer(i),Layer(i-1)+1);
    pos = pos + (Layer(i)*(Layer(i-1)+1));
end 



%   Detailed explanation goes here
m = size(X, 1);
num_labels = size(tTheta{size(Layer,1)-1}, 1);

a = cell(size(Layer,1),1);
z = cell(size(Layer,1),1);
t = cell(size(Layer,1),1);
a{1} = [ones(m, 1) X]';
for i = 2:size(Layer,1)
    z{i} = tTheta{i-1}*a{i-1};
    a{i} = [ones(1,size(X,1)) ; ActiveFunction(z{i})];
end

%{
z2 = Theta1*a1;
a2 = [ones(1,size(X,1)) ; sigmoid(z2)];
z3 = Theta2*a2;
a3 = sigmoid(z3);
%}

%yy = transformY(y,num_labels);
n = size(y,1);
res = zeros(n,num_labels);
c = 1:n:(n*num_labels);
ty = c(y) + ( 0:n-1);
res(ty) = 1;
yy =res;


J = sum(sum(-yy'.*log(a{top}(2:end,:))-(1-yy').*log(1-a{top}(2:end,:))))./m;
for i = 1:size(Layer,1)-1
    J = J + lambda/(2*m)*sum(sum(tTheta{i}(:,2:end).*tTheta{i}(:,2:end)));
end
%J = J + lambda/(2*m)*sum(sum(Theta1(:,2:end).*Theta1(:,2:end)));
%J = J + lambda/(2*m)*sum(sum(Theta2(:,2:end).*Theta2(:,2:end)));




t{top} = (a{top}(2:end,:)' - yy)./m;
for i = top-1:-1:2
  t{i} = t{i+1}*tTheta{i}.*(a{i}.*(1-a{i}))';
end


t{top} =[zeros(size(yy,1),1) t{top}];
D = cell(size(Layer,1)-1,1);
grad = [];
for i = top-1:-1:1
   %D{i} = (a{i}*t{i+1}(:,2:end))' + lambda*(tTheta{i})./m;
   %D{i}(:,1) =   D{i}(:,1) - lambda*(tTheta{i}(:,1))./m;
   D{i} = (a{i}*t{i+1}(:,2:end))' ;
   grad = [D{i}(:);grad];
end
%D1 = (Xt'*t2(:,2:end))' + lambda*(Theta1);
%D1(:,1) =   D1(:,1) - lambda*(Theta1(:,1));
%grad = [D1(:);D2(:)];

end

