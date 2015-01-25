function err = gradcheck( theta,data,y,model,lambda,eps)
%GRADCHECK Summary of this function goes here
%   Detailed explanation goes here
[J, grad]=CostFunction(theta,data,y,model,lambda);

n = size(grad,1);
err = zeros(n,1);
for i = 1: n
    if mod(i,1000)==0
        disp(i);
        disp([ min(err) max(err)  ]);
        disp(err(i-1));
    end
    tem = theta;
    theta(i)= theta(i)+eps;
    [J1, grad1]=CostFunction(theta,data,y,model,lambda);
    theta(i)= theta(i)-2*eps;
    [J2, grad2]=CostFunction(theta,data,y,model,lambda);

    err(i)= abs((J1-J2)/(2*eps)-grad(i));
    if err(i)>=eps 
        disp(err(i));
    end;
    theta = tem;
end

disp(mean(err));
disp(std(err));
end

