n = 5;
t_A = rand(n);
A = t_A*t_A';
t_B = rand(n);
B = t_B*t_B';

eig(inv((A+B)/2-(inv(A)+inv(B))/2))
