%--------------------------------------------------------
function [xs,ys,Xs,Ys] = partitionData(x,y,X,Y,M,partitionCriterion)
% Random or disjoint partition of data
% x, y - normalized training data
% X, Y - original training data
% M    - number of subsets

[n,d] = size(x) ;
if M > n; warning('The partition number M exceeds the number of training points.'); end

switch partitionCriterion
    case 'random' % 1 communication subset + (M-1) subsets
        n_per = floor(n/M) ;
        Indics = randperm(n) ;
        
        for i = 1:M
            index = Indics(1:n_per) ;
            Indics(1:n_per) = [] ;
            xs{i} = x(index,:) ; ys{i} = y(index) ;
            Xs{i} = X(index,:) ; Ys{i} = Y(index) ;
        end
        
        % assign remaining points randomly to subsets
        if length(Indics) > 0
            todo_id = randperm(M) ;
            for i = 1:length(Indics)
                xs{todo_id(i)} = [xs{todo_id(i)};x(Indics(i),:)] ; ys{todo_id(i)} = [ys{todo_id(i)};y(Indics(i))] ;
                Xs{todo_id(i)} = [Xs{todo_id(i)};X(Indics(i),:)] ; Ys{todo_id(i)} = [Ys{todo_id(i)};Y(Indics(i))] ;
            end
        end
        
    case 'kmeans' % 1 communication subset + (M-1) local subsets
        % communication subset
        n_per = floor(n/M) ;
        Indics = randperm(n) ;
        I_com = Indics(1:n_per) ;
        xs{1} = x(I_com,:) ; ys{1} = y(I_com) ;
        Xs{1} = X(I_com,:) ; Ys{1} = Y(I_com) ;
        
        % (M-1) local subsets
        Indics(1:n_per) = [] ;
        x_rest = x(Indics,:) ; y_rest = y(Indics) ;
        X_rest = X(Indics,:) ; Y_rest = Y(Indics) ;
        opts = statset('Display','off');
        [idx,C] = kmeans(x_rest,M-1,'MaxIter',500,'Options',opts);
        
        for i = 1:(M-1)
            xs{i+1} = x_rest(idx==i,:) ; ys{i+1} = y_rest(idx==i,:) ;
            Xs{i+1} = X_rest(idx==i,:) ; Ys{i+1} = Y_rest(idx==i,:) ;
        end
        
    otherwise
        error('No such partition criterion.') ;
end
end