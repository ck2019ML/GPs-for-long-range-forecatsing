%% 1. global experiment setting

opt_sm = -1000;   % the optimization times for GP
NormX = 1; NormY = 0;  % normalizing X or Y
Q=10;       % initial number of components in spectral kernels

numinit = 3;    % initialization times of each spectral kernel, a large number is preferred
sn = 0.5;        % initial signal variance
csvStr = {'airline.mat', 'monthly-electricity.mat', 'pole_telecom.mat', 'rail-miles.mat'};  % datasets

ith_data = 1;  % the i-th data used in current experiment
load(csvStr{ith_data})
disp('Load saved mat files');

[xn, xd] = size(xtrain);
xtrain_mean = zeros(1, xd); xtrain_std = ones(1, xd);
ytrain_mean = 0; ytrain_std = 1;

% normalizing data
if NormX==1 && xd>1, xtrain_mean = mean(xtrain); xtrain_std = std(xtrain); end
if NormY==1
    ytrain_mean = mean(ytrain);
    ytrain_std = std(ytrain);
end

x_norm = @(xx) (xx - xtrain_mean) ./ xtrain_std;
y_norm = @(yy) (yy - ytrain_mean) ./ ytrain_std;

x_unorm = @(xx) xx .* xtrain_std + xtrain_mean + x_start;
y_unorm = @(yy) yy .* ytrain_std + ytrain_mean;
yVar_unorm = @(yVar) yVar .* (ytrain_std^2);

inputdata.xtrain = x_norm(xtrain);
inputdata.xtest = x_norm(xtest);
inputdata.xfull = x_norm(xfull);

inputdata.ytrain = y_norm(ytrain);
inputdata.ytest = ytest;
inputdata.yfull = yfull;

inputdata.x_unorm = x_unorm;
inputdata.y_unorm = y_unorm;
inputdata.yVar_unorm = yVar_unorm;

%% 2. run benchmark kernels, such as SM, LKP, LSM
hypSM.cov = 0;
hypSM.lik = log(sn);

disp('Initialization of SM')
covfunc = {@covSM,Q};
randinit(numinit, opt_sm, covfunc, hypSM, Q, inputdata, 1);

disp('Initialization of LKP')
LKP_MC = 1;

if LKP_MC==0
    covfunc = {@covLKP, Q};
    randinit(numinit, opt_sm, covfunc, hypSM, Q, inputdata, 2);
    
else
    % test LKP from LKP script
    [mLKP_tmp, VarLKP_tmp, NLML_LKP, lkphyp] = test_lkp(Q, inputdata);
    
    ytrain = inputdata.y_unorm(inputdata.ytrain);
    mLKP = inputdata.y_unorm(mLKP_tmp);
    VarLKP = inputdata.yVar_unorm(VarLKP_tmp);
    ytest = inputdata.ytest;
    
    MAE_test = roundn(mae(mLKP, ytest), -3);
    MSE_test = roundn(immse(mLKP, ytest), -3);
    SMSE_test = MSE_test / var(ytest);
    MSLL_test = msll(ytrain, ytest, mLKP, VarLKP, lkphyp.lik);
    
    testResults.NLML_opt = NLML_LKP;
    testResults.MAE_test = MAE_test;
    testResults.MSE_test = MSE_test;
    testResults.SMSE_test = SMSE_test;
    testResults.MSLL_test = MSLL_test;
end

disp('Initialization of ALSM')
covfunc = {@covALSM, Q};
randinit(numinit, opt_sm, covfunc, hypSM, Q, inputdata, 3);
