function [hyp_opt, NLML_opt, mPred, VarPred] = InfGPmodel(hyps, covfunc, lik, opt_num, xtrain, ytrain, xtest)

[xn, xD] = size(xtrain);

if xn>=1000
    % train, Y is yes for normalizing
    BCMopts.Xnorm = 'N' ; BCMopts.Ynorm = 'N' ; partitionCriterion = 'random' ;
    BCMopts.Ms = floor(xn / 200); BCMopts.partitionCriterion = partitionCriterion ;
    BCMopts.cov = hyps.cov ; BCMopts.covfunc = covfunc;
    % opts.covfunc = @covSM;
    
    BCMopts.likfunc = lik; BCMopts.inffunc = @infGaussLik; % @infGaussLik ;
    BCMopts.numOptFC = opt_num; BCMopts.sn = hyps.lik; BCMopts.meanfunc = [];
    
    criterion = 'RBCM' ;
    [models, NLML_opt, hyp_opt, ~] = aggregation_train(xtrain, ytrain, BCMopts) ;
    if nargin>6
        [mPred, VarPred, ~] = aggregation_predict(xtest, models, criterion) ;
    else
        mPred = 0;
        VarPred = 0;
    end
    
else
    hyp_opt = minimize(hyps, @gp, opt_num, @infExact, [], covfunc, lik, xtrain, ytrain);
    NLML_opt = gp(hyp_opt, @infExact, [], covfunc, lik, xtrain, ytrain);
    if nargin>6
        [mPred, VarPred] = gp(hyp_opt, @infExact, [], covfunc, lik, xtrain, ytrain, xtest);
    else
        mPred = 0;
        VarPred = 0;
    end
end
end