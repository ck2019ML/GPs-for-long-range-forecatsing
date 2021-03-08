function [models, nlZ, hyp_opt, t_train] = aggregation_train(X,Y,opts)
% Aggeration GP for large scale training data
%
% Inputs:
%        X: a n*d matrix comprising n d-dimensional training points
%        Y: a n*1 vector containing the function responses of n training points
%        opts: options to build distributed GP models
%             .Xnorm: 'X' normalize X along each coordiante to have zero mean and unite variance
%             .Ynorm: 'Y' normalize Y to have zero mean and unite variance
%             .Ms: number of experts
%             .partitionCriterion: 'random', 'kmeans'
%             .meanfunc, .covfunc, .likfunc, .inffunc: GP configurations
%             .ell, .sf2, .sn2: parameters for the SE covariance function
%             .numOptFC: optimization setting for min-NLML
% Outputs:
%         models: a cell structure that contains the sub-models built on subsets, where models{i} is the i-th model
%                 fitted to {Xi and Yi} for 1 <= i <= M
%         t_train: computing time for min-NLML optimization
%
% H.T. Liu 2018/06/01 (htliu@ntu.edu.sg)

% Normalize training data
[n,d] = size(X) ;
x_train_mean = zeros(1,d) ; x_train_std  = ones(1,d) ;
y_train_mean = 0 ; y_train_std  = 1 ;

if strcmp(opts.Xnorm,'Y'); x_train_mean = mean(X) ; x_train_std  = std(X) ; end
x_train = (X-repmat(x_train_mean,n,1)) ./ repmat(x_train_std,n,1) ;

if strcmp(opts.Ynorm,'Y'); y_train_mean = mean(Y) ; y_train_std  = std(Y) ; end
y_train = (Y-y_train_mean)/y_train_std ;

% Partition training data into M subsets
M = opts.Ms;
[x_trains, y_trains, Xs, Ys] = partitionData(x_train,y_train,X,Y,M,opts.partitionCriterion) ;

% Infer hyperparameters by a PoE (product-of-experts) model
meanfunc = opts.meanfunc ; covfunc  = opts.covfunc ;
likfunc  = opts.likfunc ; inffunc  = opts.inffunc ;

% ell = opts.ell ;
% sf2 = opts.sf2 ;
% sn2 = opts.sn2;

hyp = struct('mean', [], 'cov', opts.cov, 'lik', opts.sn);
numOptFC = opts.numOptFC ;

t1 = clock ;
hyp_opt = minimize(hyp, @gp_factorise, numOptFC, inffunc, meanfunc, covfunc, likfunc, x_trains, y_trains);
t2 = clock ;
t_train = etime(t2,t1);

nlZ = gp_factorise(hyp_opt, inffunc, meanfunc, covfunc, likfunc, x_trains, y_trains);

% Export models
for i = 1:M
    % different for the M GP experts
    model.X = Xs{i} ; model.Y = Ys{i} ;
    model.X_norm = x_trains{i} ; model.Y_norm = y_trains{i} ;
    % same for the M GP experts
    model.hyp = hyp_opt ;
    
    model.X_mean = x_train_mean ; model.X_std = x_train_std ;
    model.Y_mean = y_train_mean ; model.Y_std = y_train_std ;
    
    model.meanfunc = meanfunc ; model.covfunc = covfunc ;
    model.likfunc = likfunc ; model.inffunc = inffunc ;
    
    model.optSet = opts;
    model.Ms = opts.Ms ;
    models{i} = model ;
end
end