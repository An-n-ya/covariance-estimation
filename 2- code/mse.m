function res = mse(A,B)
%     res = sum(A.^2)/size(A,1);
    res = norm(A-B,'fro')/norm(B,'fro');
end