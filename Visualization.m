load dataT

%x = 1:size(J,2);
x= 1:6000;
J = J(:,x);
cor=cor(x);
plotyy(x,J,x,cor);
