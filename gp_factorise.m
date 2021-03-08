function [nlZ,dnlZ] = gp_factorise(hyp, inf, mean, cov, lik, xs, ys)
% Factorised NLML
% -logp(Y|X,theta) = -\sum_{k=1}^M logp_k(Y^k|X^K,theta)

M = length(xs) ; d = size(xs{1},2) ;
nlZ = zeros(M,1) ;
dnlZ.mean = [] ; cov_grad = zeros(numel(hyp.cov),M) ; lik_grad = zeros(M,1) ;

for i = 1:M
    x = xs{i} ; y = ys{i} ;
    [nlZ_i,dnlZ_i] = gp(hyp,inf,mean,cov,lik,x,y) ;
    nlZ(i) = nlZ_i ;
    cov_grad(:,i) = dnlZ_i.cov ; lik_grad(i) = dnlZ_i.lik ;
end

nlZ = sum(nlZ);
dnlZ.cov = sum(cov_grad,2);
dnlZ.lik = sum(lik_grad);
end