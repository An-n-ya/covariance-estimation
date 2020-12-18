
A = [1 2;2 8];
B = [3 1;1 4];
C= [8,3;5,9];
S = oneinterpol(A,B,0.5)
x = naturaldist(A,B)

[t_star, value] = mindistance(A,B,C)


function S=oneinterpol(A,B,t)
 [U_A,V_A]=eig(A);
 A_half_half=U_A*diag(sqrt(sqrt(diag(V_A))));
 A_half=A_half_half*A_half_half';
 A_minus_half=U_A*diag(1./sqrt(diag(V_A)))*U_A';
 L=chol(B);
 LL=L*A_minus_half;
 [U,V]=eig(LL'*LL);
 S_half=A_half*U*diag(diag(sqrt(V)).^(t));
 S=S_half*S_half';
end
 
function S=twointerpol(A,B,C,t,s)
 P=oneinterpol(A,B,t);
 S=oneinterpol(P,C,s);
end

function x=naturaldist(A,B)
 e=eig(A,B);
 x=sqrt(sum((log(e)).^2));
end

function [t_star,value]=mindistance(A,B,C)
 fun = @(x)naturaldist(C,oneinterpol(A,B,x));
 x0 = .5; % initial point
 options = optimset('TolX',1e-15,'MaxFunEvals',100000);
 t_star = fminsearch(fun,x0,options);
 value=fun(t_star);
end