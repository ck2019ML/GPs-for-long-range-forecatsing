function [K,dK] = covALSM(Q, hyp, x, z)

% Laplace Spectral Mixture covariance function. The covariance function
% parametrization depends on the sign of Q.
% Copyright (c) by Kai Chen

if nargin<1, error('You need to provide Q.'), end
smp = Q<0; Q = abs(Q);                    % switch between covLSM and covLSMP mode
if nargin<3                                            % report no of parameters
    if smp, K = '3*D*'; else K = '(1+3*D)*'; end, K = [K,sprintf('%d',Q)]; return
end
if nargin<4, z = []; end                                   % make sure, z exists

% Note that we have two implementations covLSMRQ and covLSMfast. The former
% constructs a weighted sum of products of covGabor covariances using covMask,
% covProd, covScale and covSum while the latter is a standalone direct
% implementation. The latter tends to be faster.
if nargout > 1
    if smp
        [K,dK] = covALSMRQ(Q,hyp,x,z,smp);
    else
        [K,dK] = covALSMfast(Q,hyp,x,z,smp);               % faster direct alternative
    end
else
    K = covALSMfast(Q,hyp,x,z,smp);                      % faster direct alternative
end

function [K,dK] = covALSMfast(Q,hyp,x,z,smp)
xeqz = isempty(z); dg = strcmp(z,'diag');            % sort out different types
[n,D] = size(x); P = smp*D+(1-smp);                  % dimensionality, P=D or P=1

w = exp(reshape(  hyp(            1:P*Q), P, Q));      % Laplace mixture weights
m = exp(reshape(  hyp(P*Q+      (1:D*Q)), D, Q));      % Laplace means
sk= exp(reshape(  hyp(P*Q+D*Q+  (1:D*Q)), D, Q));      % log spectral skewness
v = exp(reshape(2*hyp(P*Q+2*D*Q+(1:D*Q)), D, Q));      % Laplace variances

if dg
    T = zeros(n,1,D);
else
    if xeqz
        T = bsxfun(@minus,reshape(x,n,1,D), reshape(x,1,n,D));
    else
        T = bsxfun(@minus,reshape(x,n,1,D), reshape(z,1,[],D));
    end
end, T = reshape(T,[],D);

Cau = 1 + (T.*T)*v./2;    % the inverse Cauchy part
sktau = T*sk;
Skew = (sktau).^2./Cau;    % the skewness part
La = cos(T*m) ./ (Cau + Skew);    % For LSM kernel, we use angular frequency !!!
K = reshape(La*w', n, []);    % the final covariance matrix

if nargout>1
    vec = @(x) x(:); % here R(:) is the derivate of hyp
    dKdhyp = @(R) [(La'*R(:)).*w';
        -vec(((sin(T*m)./(Cau + Skew).*(R(:)*w))'*                   T)'.*m);
        -vec(((2.*sktau.*La./(Cau + Skew)./Cau.*(R(:)*w))'*          T)'.*sk);
        -vec(((La./(Cau + Skew).*(1-Skew./(Cau.^2)).*(R(:)*w))'*(T.*T))'.*v)];
    dK = @(R) dirder_fast(R, T, Cau, Skew, La, dKdhyp, x, z, m, sk, v, w);
end

function [dhyp,dx] = dirder_fast(R, T, Cau, Skew, La, dKdhyp, x, z, m, sk, v, w)
%   dirder_fast(R, T, Cau, sktau, tri_part, dKdhyp, x, z, m, sk, v, w)
dhyp = dKdhyp(R);
if nargout>1
    xeqz = isempty(z); dg = strcmp(z,'diag'); [n,D] = size(x);
    if dg
        dx = zeros(size(x));
    else
        A = reshape((R(:)*w).*cos(T*m)./(Cau+Skew)*m' + ...
            (R(:)*w).*La./(Cau + Skew).*(T*v + 2*T*(sk.^2)./Cau - (T*v).*Skew./Cau), n, [], D);
        
        dx = -squeeze(sum(A, 2));
        if xeqz, dx = dx + squeeze(sum(A,1)); end
    end
end