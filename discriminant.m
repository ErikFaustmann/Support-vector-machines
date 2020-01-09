function d=discriminant(alpha,X, t, x1, x2, w0, kernelfct)
n=900;
sigma=0.01;
d=0;

%Calculation of discriminant function
for i=1:n
    ed=[X(1,i)-x1,X(2,i)-x2];
    d=d+alpha(i)*t(i)*exp(-(ed*ed')/sigma);
end
d=d+w0;

end