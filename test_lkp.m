function [mLKP, sLKP, NLML_LKP, lkphyp_opt] = test_lkp(Q, inputdata)

%% 2. Load up data

xtrain = inputdata.xtrain;
ytrain = inputdata.ytrain;
xtest = inputdata.xtest;
ytest = inputdata.ytest;

ntrain = numel(xtrain);
ntest = numel(xtest);
ntotal = ntrain + ntest;
X = xtrain;
N = length(X);
y_clean = ytrain;

%% 3. De-mean training data
% mean_trend = @(x)[ones(size(x,1)] to subtract sample mean
% mean_trend = @(x)[ones(size(x,1),x] to subtract linear trend
% mean_trend = @(x)[ones(size(x,1),x,x.^2] to subtract quadratic trend

mean_trend = @(x)[ones(size(x,1),1)];
% mean_trend = @(x)[ones(size(x,1),1),x];
mean_param = regress(ytrain,mean_trend(xtrain));
mean_fcn = @(x)mean_trend(x)*mean_param;
y = ytrain - mean_fcn(xtrain);

%% 4. Hyperparameter initialization

%% 4a. Set initial number of basis functions
J_0 = Q;

%% 4b. Find good initialization by fitting Gaussian mixture on empirical spectrum
hypinit = initSMhypersadvanced(J_0,xtrain,y,1);

init_wgt = exp(hypinit(1:J_0));
init_chi = exp(hypinit(J_0+1:2*J_0));
init_lambda_g = exp(-hypinit(2*J_0+1:end));
init_beta_g = init_wgt*var(y)/ntrain;

% Convert Gaussian parameters to Laplace parameters via least squares
init_beta = zeros(J_0,1);
init_lambda = zeros(J_0,1);
for j = 1:J_0
    xfit = linspace(-3/init_lambda_g(j),3/init_lambda_g(j),1000);
    ygauss = init_beta_g(j)*init_lambda_g(j)/sqrt(2*pi).*exp(-0.5*init_lambda_g(j)^2*xfit.^2);
    lapl = @(p)sum((p(1)*p(2)/2*exp(-p(2)*abs(xfit))-ygauss).^2);
    optimoptions('fminunc','Algorithm','quasi-newton');
    p = fminunc(lapl,[init_beta_g(j); init_lambda_g(j)],optimset('Display','none'));
    init_beta(j) = p(1);
    init_lambda(j) = p(2);
end

%% 4c. Initial spectrum plot
M = floor(ntrain/2);
freq = [[0:M],[-M+1:1:-1]]'/ntrain; 
freq = freq(1:M+1);
emp_spect = 2*abs(fft(y)).^2/ntrain;
emp_spect = emp_spect(1:M+1);
total_area = trapz(freq,emp_spect);

freq2 = (0:0.00001:0.5)';
figure(10); clf
plot(freq,emp_spect); 
hold on; 
plot(freq2,Gaussian_BF(freq2,[init_chi, init_lambda_g.^2])*init_beta_g,'m');
plot(freq2,Laplace_BF(freq2,[init_chi, init_lambda],0)*init_beta,'k');
title('Initial Spectrum')
legend('Empirical','Gaussian','Laplace')
hold off

%% 4d. Flags for whether to run MCMC on the log of parameters
%  - log scale allows for greater variation in parameter sizes
betalogscale = 1;
lambdalogscale = 1;

%% 4e. Tune Levy process hyperparameters based on initial spectrum parameters
if betalogscale == 1
    input_beta = log(init_beta);
else
    input_beta = init_beta;
end
etaparams = gamfit(abs(input_beta));
if lambdalogscale == 1
    input_lambda = log(init_lambda);
else
    input_lambda = init_lambda;
end
lamparams = gamfit(abs(input_lambda));
eps = 0.0041;      % Levy process truncation bound for small beta
alpha = 1;         % a-Stable parameter

% Tune hyperhyperparameters to center on current hyperparameters
% a_gam/b_gam*(xmax-xmin)*expint(eps) equals avg number of kernels
% a_eta/b_eta ~ avg of beta or avg of 1/eta
% a_lam/b_lam ~ avg of lambda
domain = [0, 0.5]; % Frequency domain
a_gam_0 = 2.53;
b_gam_0 = 6.45*(domain(2)-domain(1))*expint(eps)/25;
a_eta_0 = etaparams(1);
b_eta_0 = 1/etaparams(2);
a_lam_0 = lamparams(1);
b_lam_0 = 0.5/lamparams(2);
gam_0 = a_gam_0/b_gam_0;
eta_0 = 1/(a_eta_0/b_eta_0);

% Struct for Initial Levy Prior
LevyPrior_0.name = 'Gamma'; % 'Gamma', 'SymGamma', or 'Stable'
LevyPrior_0.hyperhyperparams = [a_gam_0; b_gam_0; a_eta_0; b_eta_0];
LevyPrior_0.hyperparams = [gam_0; eta_0; eps; alpha];
LevyPrior_0.betalogscale = betalogscale;

% Struct for Basis Function and its parameters
BasisFunction.domain = [domain(1) - 0.001, domain(2) + 0.001];
BasisFunction.function = @(x, BFParams) Laplace_BF(x, BFParams, lambdalogscale);
BasisFunction.function_ift = @(x, BFParams) Laplace_BF_ift(x, BFParams, lambdalogscale);
BasisFunction.hyperparams = [a_lam_0, b_lam_0];
BasisFunction.lambdalogscale = lambdalogscale;

%% 4f. Summary to aid in Hyperparameter Tuning
fprintf('\nPrior Hyperparameter Tuning:\n')
fprintf('Average # Basis Functions: %.5f\n',a_gam_0/b_gam_0*(domain(2)-domain(1))*expint(eps))
if betalogscale == 1
    fprintf('Average of log(Beta): %.5f\n',a_eta_0/b_eta_0)
else
    fprintf('Average of Beta: %.5f\n',a_eta_0/b_eta_0)
end
if lambdalogscale == 1
    fprintf('Average of log(Lambda): %.5f\n\n',a_lam_0/b_lam_0)
else
    fprintf('Average of Lambda: %.5f\n\n',a_lam_0/b_lam_0)
end

%% 5. Setup and call RJ-MCMC

%% 5a. RJ-MCMC parameter tuning
% Noise Variance
sigma2 = 0.001;

% Number of RJ-MCMC Samples
% numSamples = 5000;
% numSamples = 2500;
% numSamples = 1000;
numSamples = 500;
% numSamples = 250;
% numSamples = 100;
% numSamples = 2;

% Metropolis-Hastings Proposal Step Sizes for
% [ (log)beta, chi, (log)lambda, gamma, eta]
if lambdalogscale == 1
    proposalStepSize = [1/eta_0, 0.2, mean(log(init_lambda)), 0.2*gam_0, 0.2*eta_0];
else
    proposalStepSize = [1/eta_0, 0.2, mean(init_lambda), 0.2*gam_0, 0.2*eta_0];
end

% RJ-MCMC Move Type Probabilities
birthProb = 0.1;
deathProb = 0.1;
updateProb = 1-birthProb-deathProb;

% Proportion of RJ-MCMC steps for hyperparameter updates
hyperUpdateProb = 0.15; 
MoveProb = [birthProb, deathProb, updateProb, hyperUpdateProb];

% Initial RJ-MCMC Parameters
theta_0 = zeros(3,J_0);
if betalogscale == 1
    theta_0(1,:) = log(init_beta);
else
    theta_0(1,:) = init_beta;
end
theta_0(2,:) = init_chi;
if lambdalogscale == 1
    theta_0(3,:) = log(init_lambda);
else
    theta_0(3,:) = init_lambda;
end

%% 5b. Parameters for Structured Kernel Interpolation (SKI):
useSKI = 0; % Set to 1 to use SKI. Set to 0 for exact covariance

% i. Wrapper for the inverse Fourier transform of the basis function to make it compatible with GPML
SKIParams.bf_ift_wrapper = @Laplace_BF_ift_wrapper;

% ii. Grid of Inducing Points for SKI
% If training data is small enough ( < O(10^4) ) then we can simply supply the whole training dataset as inputs.
% SKIParams.xg = [xtrain; xtest];                             % SKI grid points
% If training data is large, then supply number of grid points to SKIParams.ng and determine the grid based on training and testing inputs
SKIParams.ng = 100;                                         % Number of SKI grid points
SKIParams.xg = linspace(min([xtrain; xtest]),max([xtrain; xtest]),SKIParams.ng)';   % SKI grid points

% iii. Conjugate Gradient Parameters - If SKI consistently fails to converge or is too slow, adjust these accordingly
SKIParams.cg_maxit = 10000;                                  % Max Iterations for Conjugate Gradient in SKI
SKIParams.cg_tol = 1e-8;                                    % Residual Tolerance for Conjugate Gradient in SKI
SKIParams.cg_showit = 1;                                    % Display number of CG iterations to convergence (1 to activate)

%% 5c. Call RJ-MCMC to sample Levy Kernel Process posterior
% samples = struct containing sampled kernel parameters
% accept = matrix containing types of steps and indicators for acceptance or rejection
tic;
if useSKI == 1
    [samples, accept] = Levy_RJMCMC_Sampler(y, X, J_0, theta_0, numSamples, proposalStepSize, LevyPrior_0, BasisFunction, sigma2, MoveProb, SKIParams);
else
    [samples, accept] = Levy_RJMCMC_Sampler(y, X, J_0, theta_0, numSamples, proposalStepSize, LevyPrior_0, BasisFunction, sigma2, MoveProb);
end
T_RJMCMC = toc;

fprintf('LARK RJMCMC finished in %g seconds\nJ_Final = %g\n', T_RJMCMC, samples.J(end))

%% 5d. Compute acceptance probabilities

% Used to diagnose the mixing of the MCMC run.
update_steps = (accept(:,1) == 2);
update_hyp_steps = (accept(:,1) == 4);
birth_steps = (accept(:,1) == 1);
death_steps = (accept(:,1) == 3)|(accept(:,1) == -1);

update_acceptrate(1) = sum(accept(update_steps,2))/sum(update_steps);
update_acceptrate(2) = sum(accept(update_steps,3))/sum(update_steps);
update_acceptrate(3) = sum(accept(update_steps,4))/sum(update_steps);
update_acceptrate(4) = sum(accept(update_hyp_steps,5))/sum(update_hyp_steps);
update_acceptrate(5) = sum(accept(update_hyp_steps,6))/sum(update_hyp_steps);
birth_acceptrate = sum(accept(birth_steps,2)==1)/sum(birth_steps);
death_acceptrate = sum(accept(death_steps,2)==1)/sum(death_steps);

fprintf('Acceptance probabilities:\nbeta   = %g \nchi    = %g\nlambda = %g\ngamma  = %g\neta    = %g\nbirth  = %g\ndeath  = %g\n\n', [update_acceptrate, birth_acceptrate, death_acceptrate])

%% 6. Estimate Credible Interval of Predictive Distribution

% Input locations to predict
Xstar = [xtrain+0.5*(xtrain(2)-xtrain(1)); xtest];
Nstar = length(Xstar);

% Number of sample kernels to use
nPredict = 500;
% nPredict = 250;
% nPredict = 10;
% nPredict = 1;

if nPredict > numSamples
    nPredict = numSamples;
end

% 1 to use SKI, 0 for exact covariance
useSKI = 0; 
if useSKI == 0
    fprintf('Calculating predictive distribution with exact covariances\n')
else
    fprintf('Calculating predictive distribution with SKI approximation\n')
end

% 1 to estimate 2.5% and 97.5% quantiles of predictive distribution
% 0 to estimate mean and variance of predictive distribution
computeQuantile = 1;

% Quantile calculation not available for SKI. Also cannot compute quantiles
% if there are not enough samples.
if useSKI == 1 || nPredict < 100
    computeQuantile = 0; 
end

if computeQuantile == 1
    fgf = zeros(Nstar,nPredict);
    Zrand = randn(N+Nstar,nPredict);
    samp_idx = 1;
end
fmugf = zeros(Nstar,1);
fs2gf = zeros(Nstar,1);
ymugf = zeros(Nstar,1);
ys2gf = zeros(Nstar,1);

% Compute GP Predictions over the last nPredict Random Kernels
for s = numSamples-nPredict+1:numSamples
    fprintf('Predictive Sample %d\n',s-numSamples+nPredict)
    if nPredict == 1
        % If only one element for Bayes Average, take the MAP estimate
        [~,s] = max(real(samples.log_Posterior));
    end
    J_s = samples.J(s);
    JJ = [0; cumsum(samples.J*3)];
    theta = reshape(samples.theta(JJ(s)+1:JJ(s+1)), 3, J_s);
    BFParams = theta(2:end,:)';
    
    tic
    % % Recover optimal kernel
    if useSKI == 1
        % Use SKI to approximate kernel (for large training data sets)
        gpcov = {SKIParams.bf_ift_wrapper,betalogscale,lambdalogscale,BFParams,theta(1,:)'};
        gpmean = {@meanZero};
        lik = {@likGauss};
        hyp.cov = [];
        hyp.mean = [];
        hyp.lik = 0.5*log(sigma2);
        covg = {@apxGrid,{gpcov},{SKIParams.xg}};% grid prediction

        opt.cg_maxit = 20000;
        opt.cg_tol = 1e-8;
        opt.cg_showit = 1;
        opt.use_pred_var = 1;
        if opt.use_pred_var == 1
            opt.pred_var = 100;
        elseif isfield(opt,'pred_var')
            opt = rmfield(opt,'pred_var');
        end
        [postg,nlZg] = infGrid(hyp,gpmean,covg,lik,X,y,opt);  % fast grid prediction
        [fmugf_s,fs2gf_s,ymugf_s,ys2gf_s] = postg.predict(Xstar);
        NLML_LKP = nlZg;
        
    else
        % Compute exact covariance (for small training data sets)
        if betalogscale == 1
            beta = exp(theta(1,:)');
        else
            beta = theta(1,:)';
        end
        
        % Check if input locations are equally spaced
        eqspaceX = range(diff(X)) < 1e-8;
        if eqspaceX
            tauX = linspace(0,(X(2)-X(1))*(N-1),N)';
        else
            tauX = repmat(X,[1,N])-repmat(X',[N,1]);
        end
        eqspaceXstar = range(diff(Xstar)) < 1e-8;
        if eqspaceXstar
            tauXstar = linspace(0,(Xstar(2)-Xstar(1))*(Nstar-1),Nstar)';
        else
            tauXstar = repmat(Xstar,[1,Nstar])-repmat(Xstar',[Nstar,1]);
        end
        
        % Apply kernel to all training and test input pairs
        
        % If input locations are equally spaced, apply Toeplitz shortcut
        if eqspaceX
            K_Synth_xx = toeplitz(BasisFunction.function_ift(tauX, BFParams)*beta);
        else
            K_Synth_xx = reshape(BasisFunction.function_ift(tauX(:), BFParams)*beta,N,N);
        end
        if eqspaceXstar
            K_Synth_zz = toeplitz(BasisFunction.function_ift(tauXstar, BFParams)*beta);
        else
            K_Synth_zz = reshape(BasisFunction.function_ift(tauXstar(:), BFParams)*beta,Nstar,Nstar);
        end
        tauXXstar = repmat(Xstar,[1,N])-repmat(X',[Nstar,1]);
        K_Synth_zx = reshape(BasisFunction.function_ift(tauXXstar(:), BFParams)*beta,Nstar,N);
        fmugf_s = K_Synth_zx * ((K_Synth_xx + sigma2*eye(ntrain)) \ (y));
        fcovgf_s = K_Synth_zz - K_Synth_zx * ((K_Synth_xx + sigma2*eye(ntrain)) \ K_Synth_zx');
        fs2gf_s = diag(fcovgf_s);
        ymugf_s = fmugf_s;
        ys2gf_s = fs2gf_s + sigma2;
    end
    toc
    
    % Posterior predictive
    % ystar = fstar + eps,  eps ~ N(0,sigma^2)
    % fstar|X,y ~ N(fmugf,fs2gf)
    % ystar|X,y ~ N(ymugf,ys2gf)
    
    % If quantiles are to be calculated, then compute a sample GP function
    if computeQuantile == 1
        % Do Cholesky on unconditional matrix to avoid non-PSD matrices
        C_Synth = cholcov([K_Synth_xx, K_Synth_zx'; K_Synth_zx, K_Synth_zz])';
        uncond = C_Synth*Zrand(1:size(C_Synth,2),samp_idx);
        fgf(:,samp_idx) = mean_fcn(Xstar) + K_Synth_zx*(K_Synth_xx\(y-uncond(1:N)))+uncond(N+1:end);
        samp_idx = samp_idx+1;
    end
    fmugf = fmugf + 1/nPredict*fmugf_s;
    fs2gf = fs2gf + 1/nPredict*fs2gf_s;
    ymugf = ymugf + 1/nPredict*ymugf_s;
    ys2gf = ys2gf + 1/nPredict*ys2gf_s;
end
E_fstar_Synth = mean_fcn(Xstar) + ymugf;
Std_fstar_Synth_95 = sqrt(ys2gf);
if computeQuantile == 1
    fstar_lower95 = quantile(fgf,0.025,2);
    fstar_upper95 = quantile(fgf,0.975,2);
else
    fstar_lower95 = E_fstar_Synth - 2*Std_fstar_Synth_95;
    fstar_upper95 = E_fstar_Synth + 2*Std_fstar_Synth_95;
end

%% 7. Plots to Visualize and Diagnose Results

%% 7a. MAP Spectrum

% Use MAP kernel for plots
[~,s] = max(real(samples.log_Posterior));
J_s = samples.J(s);
JJ = [0; cumsum(samples.J*3)];

theta = reshape(samples.theta(JJ(s)+1:JJ(s+1)), 3, J_s);
if betalogscale == 1
    beta = exp(theta(1,:))';
else
    beta = theta(1,:)';
end
BFParams = theta(2:end,:)';
tauX=linspace(0,max(X),N)';
[k_analytical_xx] = BasisFunction.function_ift(tauX, BFParams)*beta;

% Plot MAP Spectrum
theta = reshape(samples.theta(JJ(s)+1:JJ(s+1)), 3, J_s);
X_frequency = linspace(domain(1), domain(2), 500);
bases = RJMCMC_Decoder(theta, BasisFunction, betalogscale, X_frequency ,J_s);
k_s = sum(bases, 2)+sigma2;

if lambdalogscale==1
    tmplambda = exp(theta(3, :)');
else
    tmplambda = theta(3, :)';
end
lkphyp_opt.cov = log([beta; theta(2, :)'; sqrt(2)./tmplambda]);

figure(1); clf
subplot(6,1,1:4)
hold on
plot(freq(2:end), emp_spect(2:end),'b','LineWidth',2)
plot(X_frequency, k_s, 'k', 'LineWidth', 2)
hold off
legend('Empirical Spectrum','Fitted Spectrum')
set(gca, 'yscale', 'log')
title('Power Spectral Density of Kernel')

% Plot basis functions
subplot(6, 1, 5:6)
plot(X_frequency, bases)
hold on
centers = zeros(J_s,1);
for j = 1:J_s
    [~, i] = max(abs(bases(:,j)));
    centers(j) = X_frequency(i);
    line([ X_frequency(i),  X_frequency(i)], ...
        [0, bases(i,j)],...
        'Color', 'k', 'LineStyle', ':')
end
scatter(centers, 0*ones(length(centers),1), 30, 'k', 'filled')
plot(X_frequency, bases)
hold off
title(sprintf('Basis Functions (There are %g of them)', samples.J(s)))
lkphyp_opt.lkpQ = samples.J(s);

%% 7b. Plot other diagnostics
% Plot kernel
figure(3); clf
plot(tauX, k_analytical_xx)
title('Learned Kernel')

% Plot basis parameters
figure(4); clf
stem(init_chi, init_beta, 'b'); hold on
stem(BFParams(:,1), beta, 'r'); hold off
title('Migration of beta and chi'); legend('Original', 'Optimal');
set(gca, 'yscale', 'log')

figure(5)
plot((1:numSamples)',samples.log_Posterior, 'k'); hold on
plot((1:numSamples)',samples.log_Likelihood, 'b');
plot((1:numSamples)',samples.log_Posterior-samples.log_Likelihood, 'r');
scatter(s, samples.log_Posterior(s), 'filled'); hold off
legend('log Posterior', 'log Likelihood', 'log Prior', 'Location', 'best')
title('Log Posterior')

NLML_LKP = abs(samples.log_Likelihood(end));
lkphyp_opt.lik = 0.5*log(sigma2);

%% 7c. Plot predictive distribution vs withheld test data

ymean = E_fstar_Synth(numel(xtrain)+1:end);
yCI = Std_fstar_Synth_95(numel(xtrain)+1:end);
mLKP = ymean; sLKP = yCI.^2; NLML_LKP

figure(2);clf
hold on 
ht = plot(Xstar, E_fstar_Synth, 'k');
h = area(Xstar, [fstar_upper95, E_fstar_Synth-fstar_upper95, fstar_lower95-E_fstar_Synth]);
set(h(1), 'visible', 'off')
set(h(2), 'FaceColor', 0.8*[1, 1, 1])
set(h(3), 'FaceColor', 0.8*[1, 1, 1])
hs = scatter(X, y  + mean_fcn(X), 50, 'k', 'c', 'filled');
hp = plot(X, y_clean, ':', 'LineWidth', 2);
hold off
xlabel('X'); ylabel('f(X)'); 
title('Mean Predictions: Laplace kernel')
legend([hs, ht, hp], {'Data', 'Predictive Mean', 'True Function'},'Location','Best');
hold off

hold on
plot(xtest, ytest, 'g', 'LineWidth', 2)
hold off


