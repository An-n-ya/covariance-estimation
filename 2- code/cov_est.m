rng('default')
n = 10;
m = 100;
D = diag(rand(n,1));
U = orth(rand(n,n));
R0 = U' * D * U;

X = mvnrnd(zeros(n,1),R0,m);

manifold = sympositivedefinitefactory(n);
problem.M = manifold;
problem.cost = @(R) -1/m * log(det(R)) + trace(X*R*X');
problem.grad = @(R) R*(X'*X-inv(R)/m)*R;

checkgradient(problem)

[R, xcost, info, options] = steepestdescent(problem);

inv(R)/m^2-cov(X)
inv(R)/m^2-R0

% function f = cost(R)
%         f = -1/m * log(det(R)) + trace(X'*R*X);
% end

figure;
semilogy([info.iter], [info.gradnorm], '.-');
xlabel('Iteration number');
ylabel('Norm of the gradient of f');