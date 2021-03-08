function [mu, s2, t_predict] = aggregation_predict(Xt, models, criterion)
% Aggregation GP for prediction
% Inputs:
%        Xt: a nt*d matrix containing nt d-dimensional test points
%        models: a cell structure that contains the sub-models built on subsets, where models{i} is the i-th model
%                 fitted to {Xi and Yi} for 1 <= i <= M
%        criterion: an aggregation criterion to combine the predictions from M sub-models
%                   'PoE': product of GP experts
%                   'GPoE': generalized product of GP experts
%                   'BCM': Bayesian committee machine
%                   'RBCM': robust Bayesian committee machine
%                   'GRBCM': generalized robust Bayesian committee machine
%                   'NPAE': nested pointwise aggregation of experts
% Outputs:
%         mu: a nt*1 vector that represents the prediction mean at nt test points
%         s2: a nt*1 vector that represents the prediction variance at nt test points
%         t_predict: computing time for predictions
%
% H.T. Liu 2018/06/01 (htliu@ntu.edu.sg)

nt = size(Xt,1) ;  % number of test points
M = models{1}.Ms ; % number of experts

% normalization of test points Xt
if strcmp(models{1}.optSet.Xnorm,'Y')
    Xt = (Xt - repmat(models{1}.X_mean,nt,1)) ./ (repmat(models{1}.X_std,nt,1)) ;
end

% predictions of each submodel
t1 = clock ;
if ~strcmp(criterion,'GRBCM') % no use for GRBCM
    for i = 1:M
        [mu_experts{i},s2_experts{i}] = gp(models{i}.hyp, models{i}.inffunc, models{i}.meanfunc, ...
            models{i}.covfunc, models{i}.likfunc, models{i}.X_norm, models{i}.Y_norm, Xt);
    end
end

% use an aggregation criterion to combine predictions from submodels
mu = zeros(nt,1) ; s2 = zeros(nt,1) ;
switch criterion
    case 'PoE' % product of GP experts
        for i = 1:M
            s2 = s2 + 1./s2_experts{i} ;
        end
        s2 = 1./s2 ;
        
        for i = 1:M
            mu = mu + s2.*(mu_experts{i}./s2_experts{i}) ;
        end
        
    case 'GPoE' % generalized product of GP experts using beta_i = 1/M
        for i = 1:M
            beta{i} = 1/M*ones(length(s2_experts{i}),1) ;
            s2 = s2 + beta{i}./s2_experts{i} ;
        end
        s2 = 1./s2 ;
        
        for i = 1:M
            mu = mu + s2.*(beta{i}.*mu_experts{i}./s2_experts{i}) ;
        end
        
    case 'BCM' % Bayesian committee machine
        kss = feval(models{1}.covfunc{:},models{1}.hyp.cov,Xt,'diag') + exp(2*models{1}.hyp.lik); % because s2_experts consider noise
        
        for i = 1:M
            s2 = s2 + 1./s2_experts{i} ;
        end
        s2 = 1./(s2 + (1-M)./kss) ;
        
        for i = 1:M
            mu = mu + s2.*(mu_experts{i}./s2_experts{i}) ;
        end
        
    case 'RBCM' % robust Bayesian committee machine
        kss = feval(models{1}.covfunc{:},models{1}.hyp.cov,Xt,'diag') + exp(2*models{1}.hyp.lik); % because s2_experts consider noise
        
        beta_total = zeros(nt,1) ;
        for i = 1:M
            beta{i} = 0.5*(log(kss) - log(s2_experts{i})) ;
            beta_total = beta_total + beta{i} ;
            
            s2 = s2 + beta{i}./s2_experts{i} ;
        end
        s2 = 1./(s2 + (1-beta_total)./kss) ;
        
        for i = 1:M
            mu = mu + s2.*(beta{i}.*mu_experts{i}./s2_experts{i}) ;
        end
        
    case 'GRBCM'
        % build M submodels based on cross data {X1,Xi} (1<=i<=M)
        kss = feval(models{1}.covfunc{:},models{1}.hyp.cov,Xt,'diag') + exp(2*models{1}.hyp.lik); % because s2_experts consider noise
        
        for i = 1:M
            if i == 1
                models_cross{i} = models{i} ;
            else
                model = models{i} ;
                model.X = [models{1}.X;models{i}.X] ; model.Y = [models{1}.Y;models{i}.Y] ; % X1 + Xi % Y1 + Yi
                model.X_norm = [models{1}.X_norm;models{i}.X_norm] ; model.Y_norm = [models{1}.Y_norm;models{i}.Y_norm] ;
                
                models_cross{i} = model ;
            end
        end
        
        for i = 1:M
            [mu_crossExperts{i},s2_crossExperts{i}] = gp(models_cross{i}.hyp,models_cross{i}.inffunc,models_cross{i}.meanfunc, ...
                models_cross{i}.covfunc,models_cross{i}.likfunc,models_cross{i}.X_norm,models_cross{i}.Y_norm,Xt);
        end
        
        % combine predictions from GP experts
        beta_total = zeros(nt,1) ;
        %zero_num = zeros(M,1);
        for i = 1:M
            if i > 2
                beta{i} = 0.5*(log(s2_crossExperts{1}) - log(s2_crossExperts{i})) ;
            else
                beta{i} = ones(nt,1) ; % beta_1 = beat_2 = 1 ;
            end
            beta_total = beta_total + beta{i} ;
            s2 = s2 + beta{i}./s2_crossExperts{i} ;
        end
        
        s2 = 1./(s2 + (1-beta_total)./s2_crossExperts{1}) ;
        
        for i = 1:M
            mu = mu + beta{i}.*mu_crossExperts{i}./s2_crossExperts{i} ;
        end
        mu = s2.*(mu + (1-beta_total).*mu_crossExperts{1}./s2_crossExperts{1});
        
    case 'NPAE'
        kss = feval(models{1}.covfunc{:},models{1}.hyp.cov,Xt,'diag'); % no noise
        K_invs = inverseKernelMarix_submodels(models) ;
        K_cross = crossKernelMatrix_nestedKG(models) ;
        hyp_lik = models{1}.hyp.lik ; % noise para
        
        mu_all = zeros(nt,M) ;
        for i = 1:M, mu_all(:,i) = mu_experts{i} ; end
        
        for i = 1:nt
            M_x = mu_all(i,:)' ;
            
            x = Xt(i,:) ;
            k_M_x = kernelVector_nestedKG(x,models,K_invs) ;
            [K_M_x,K_M_x_inv] = kernelMatrix_nestedKG(x,models,K_invs,K_cross) ;
            
            mu(i) = k_M_x'*K_M_x_inv*M_x ;
            s2(i) = kss(i) - k_M_x'*K_M_x_inv*k_M_x + exp(2*hyp_lik) ;
        end
        
    otherwise
        error('No such aggregation model.') ;
end

muf = mu; s2f = s2 - exp(2*models{1}.hyp.lik);

% restore predictions if needed
if strcmp(models{1}.optSet.Ynorm,'Y')
    mu = mu*models{1}.Y_std + models{1}.Y_mean ;
    s2 = s2*(models{1}.Y_std)^2 ;
    muf = muf*models{1}.Y_std + models{1}.Y_mean ;
    s2f = s2f*(models{1}.Y_std)^2 ;
end

t2 = clock ;
t_predict = etime(t2,t1) ;
end