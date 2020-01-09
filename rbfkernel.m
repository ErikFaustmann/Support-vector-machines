function k =rbfkernel(x1, x2, sigma)

n=900;
k = zeros(n,n);

for i=1:n
    for j=i:n
        ed=x1(:,i)-x2(:,j);
        k(i,j) = exp(-(ed'*ed)/sigma); 
        k(j,i)=k(i,j);
    end
end

end