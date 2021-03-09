function [testResults, restQ] = randinit(numinit, opt_num, covfunc, hyp_init, Q, inputdata, lsm_mode)
% try numinit random initialisations for SM
xtrain = inputdata.xtrain;
xtest = inputdata.xtest;
xfull = inputdata.xfull;

ytrain = inputdata.ytrain;
ytest = inputdata.ytest;
yfull = inputdata.yfull;

nlml = Inf; lik = {@likGauss};
[xn, xD] = size(xtrain);

for j=1:numinit
    
    xtrain_init = xtrain;
    ytrain_init = ytrain;
    init_flag = rem(j, 2);
    
    % init_emp_spec(Q, xtrain, ytrain, log_flag, win_flag, fft_mode, lsm_mode)
    [hyp_init.cov, temp_log_s, temp_log_spec, temp_other_log_spec] = init_emp_spec(Q, xtrain_init, ytrain_init, init_flag, lsm_mode);
    % smhyp_try = initSMhypers_init(Q,xtrain, ytrain);       % initialise SM hypers
    [hyp_init_try, nlml_new, ~, ~] = InfGPmodel(hyp_init, covfunc, lik, -200, xtrain_init, ytrain_init);
    
    if (nlml_new < nlml)
        log_s = temp_log_s;
        log_spec = temp_log_spec;
        other_log_spec = temp_other_log_spec;
        
        % get the initial value of better try
        hyp_train = hyp_init_try;
        hyp_train_start = hyp_init;
        nlml = nlml_new;
    end
end

init_w = reshape( hyp_train.cov(       1:1*Q),  1, Q);       % Laplace mixture weights
init_win = init_w>=1;
if sum(init_win)==0 || Q==1, init_win = 1; end

init_w = reshape( hyp_train_start.cov(       1:1*Q),  1, Q);       % Laplace mixture weights
init_m = reshape( hyp_train_start.cov(1*Q+(1:xD*Q)), xD, Q);      % Laplace means

w_win = init_w(init_win);
m_win = init_m(:, init_win);

if lsm_mode == 3
    init_sk= reshape(hyp_train_start.cov(1*Q+xD*Q +(1:xD*Q)), xD, Q);      % Laplace skewness
    init_s = reshape(hyp_train_start.cov(1*Q+2*xD*Q+(1:xD*Q)),xD, Q);      % Laplace variances
    sk_win = init_sk(:, init_win);
    s_win = init_s(:, init_win);
    hyp_train_win.cov = [w_win(:); m_win(:); sk_win(:); s_win(:)];
    
else
    init_s = reshape(hyp_train_start.cov(1*Q+xD*Q+(1:xD*Q)),xD, Q);      % Laplace variances from Lambda
    s_win = init_s(:, init_win);
    hyp_train_win.cov = [w_win(:); m_win(:); s_win(:)];
end

disp('Perhaps the best number of components')
restQ = sum(init_win);
covfunc{2} = restQ;
hyp_train_win.lik = hyp_train_start.lik;

[hyp_opt, NLML_opt, mPred_tmp, VarPred_tmp] = InfGPmodel(hyp_train_win, covfunc, lik, opt_num, xtrain, ytrain, xtest);
ytrain = inputdata.y_unorm(ytrain);
mPred = inputdata.y_unorm(mPred_tmp);
VarPred = inputdata.yVar_unorm(VarPred_tmp);

MAE_test = roundn(mae(mPred, ytest), -3)
MSE_test = roundn(immse(mPred, ytest), -3)
SMSE_test = MSE_test / var(ytest)

%% if xD<2, plot

xtrain = inputdata.x_unorm(xtrain);
xtest = inputdata.x_unorm(xtest);

if  xD==1
    % plot predictions
    plotNotes(tLabels, xtrain, ytrain, xtest, ytest, mPred, VarPred);
    % plot empirical density
    plotSpec(restQ, log_s, other_log_spec, hyp_opt.cov, lsm_mode)
end
end
