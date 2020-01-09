function [alpha, w0, w] = trainSVM(X,t, kernelfct)
n=900;
C=1;
sigma=0.01;
X1=X;

k=kernelfct(X,X1,sigma);

H = zeros(n,n);
for i=1:n
    for j=i:n
        H(i,j) = t(i)*t(j)*k(i,j);
        H(j,i) = H(i,j); 
    end
end

f=-ones(n,1); %das Minus Vorzeichen kommt, da quadprog eignentlich das Minimum sucht, daher müssen in der Lagrange-Funktion alle Vorzeichen umgekehrt werden

%Bedingung alpha*y == Null
Aeq=t; 
beq=0;

%Bedingung für alpha >= Nullund <=C/N
lb=zeros(n,1);

ub=(C/n)*ones(n,1);

%ub=C*ones(n,1);
alpha=quadprog(H,f,[],[],Aeq,beq,lb,ub)';

AlmostZero=(abs(alpha)<max(abs(alpha))/1e5);
alpha(AlmostZero)=0;
S=find(alpha>0 & alpha<C);

w=0;
for i=S
    
    w=w+alpha(i)*t(i)*X(:,i);
end

%Calculation of w0
z=0;

%find support vector with |d(x(S)|=1 ??????

for i=S
   ed=X(:,i)-X(:,S(2));
   z=z-alpha(i)*t(i)*exp(-(ed'*ed)/sigma);
end
w0=mean(t(S(2))+z);
end
