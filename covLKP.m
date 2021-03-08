function [K,dK] = covLKP(Q, hyp, x, z)

% Laplace Spectral Mixture covariance function. The covariance function
% parametrization depends on the sign of Q.
% Copyright (c) by Kai Chen

if nargin<1, error('You need to provide Q.'), end
smp = Q<0; Q = abs(Q);                    % switch between covLSM and covLSMP mode
if nargin<3                                            % report no of parameters
    if smp, K = '3*D*'; else K = '(1+2*D)*'; end, K = [K,sprintf('%d',Q)]; return
end
if nargin<4, z = []; end                                   % make sure, z exists

% Note that we have two implementations covLSMRQ and covLSMfast. The former
% constructs a weighted sum of products of covGabor covariances using covMask,
% covProd, covScale and covSum while the latter is a standalone direct
% implementation. The latter tends to be faster.
if nargout > 1
    if smp
        [K,dK] = covLSMRQ(Q,hyp,x,z,smp);
    else
        [K,dK] = covLSMfast(Q,hyp,x,z,smp);               % faster direct alternative
    end
else
    K = covLSMfast(Q,hyp,x,z,smp);                      % faster direct alternative
end

function [K,dK] = covLSMfast(Q,hyp,x,z,smp)
xeqz = isempty(z); dg = strcmp(z,'diag');            % sort out different types
[n,D] = size(x); P = smp*D+(1-smp);                  % dimensionality, P=D or P=1

w = exp(reshape(  hyp(         1:P*Q) , P, Q));      % Laplace mixture weights
m = exp(reshape(  hyp(P*Q+    (1:D*Q)), D, Q));      % Laplace means
v = exp(reshape(2*hyp(P*Q+D*Q+(1:D*Q)), D, Q));      % Laplace variances

if dg
    T = zeros(n,1,D);
else
    if xeqz
        T = 2*pi*bsxfun(@minus,reshape(x,n,1,D), reshape(x,1,n,D));
    else
        T = 2*pi*bsxfun(@minus,reshape(x,n,1,D), reshape(z,1,[],D));
    end
end, T = reshape(T,[],D);

RQ = 1 + (T.*T)*v./2;  % the RQ part
La = cos(T*m) ./ RQ;   % For LSM kernel, we use angular frequency !!!
K = reshape(La*w', n, []);

if nargout>1
    vec = @(x) x(:); % here R(:) is the derivate of hyp
    dKdhyp = @(R) [(La'*R(:)).*w';
        -vec(((sin(T*m)./RQ.*(R(:)*w))'*     T)'.*m );
        -vec(((La./RQ      .*(R(:)*w))'*(T.*T))'.*v )];
    dK = @(R) dirder_fast(R, RQ, T, La, dKdhyp, x, z, m, v, w);
end

function [dhyp,dx] = dirder_fast(R, RQ, T, La, dKdhyp, x, z, m, v, w)
dhyp = dKdhyp(R);
if nargout>1
    xeqz = isempty(z); dg = strcmp(z,'diag'); [n,D] = size(x);
    if dg
        dx = zeros(size(x));
    else
        A = reshape((R(:)*w).*sin(T*m)./RQ*m' + (((R(:)*w).*La./RQ)*v').*T, n, [], D);
        dx = -squeeze(sum(A,2));
        if xeqz, dx = dx + squeeze(sum(A,1)); end
    end
end