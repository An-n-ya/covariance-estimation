n = 10;
A = rand(n);
x1 = rand(n,1);
x2 = rand(n,1);
x3 = rand(n,1);

T1 = x1'*A*x1+x2'*A*x2+x3'*A*x3;

X = [x1,x2,x3];
T1-trace(X'*A*X)

B = rand(n);

