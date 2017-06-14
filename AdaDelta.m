function [train_loss, train_acc, test_loss, test_acc, train_time] ...
    = AdaDelta(x_train, y_train, x_test, y_test, opts)
% AdaDelta solver
if ~isfield(opts, 'max_iter')
    max_iter = 300;
else
    max_iter = opts.max_iter;
end
if ~isfield(opts, 'eps')
    eps = 1e-6;
else
    eps = opts.eps;
end
if ~isfield(opts, 'lamb')
    lamb = 1.;
else
    lamb = opts.lamb;
end
if ~isfield(opts, 'rho')
    rho = 0.5;
else
    rho = opts.rho;
end

rng('default'); % random seed
[~, n_features] = size(x_train);
[~, n_labels] = size(y_train);
G = zeros(n_features, n_labels);
W = zeros(n_features, n_labels);
w = randn(n_features, n_labels);
train_loss = zeros(max_iter, 1);
train_acc = zeros(max_iter, 1);
test_loss = zeros(max_iter, 1);
test_acc = zeros(max_iter, 1);
train_time = zeros(max_iter, 1);
for i = 1:max_iter
    tic;
    fprintf('iter: %d/%d\n', i, max_iter);
    g = Softgrad(y_train, w, x_train, lamb);
    G = rho*G + (1-rho)*g.^2;
    d = - sqrt(W + eps) ./ sqrt(G + eps).* g ;
    W = rho*W + (1-rho)*d.^2;
    w = w + d;
    train_time(i) = toc;
    % train eval
    [acc, loss, ~] = Softloss(y_train, w, x_train, lamb);
    train_loss(i) = loss;
    train_acc(i) = acc;
    % test eval
    [acc, loss, ~] = Softloss(y_test, w, x_test, lamb);
    test_loss(i) = loss;
    test_acc(i) = acc;
    fprintf('CPU time: %f, train loss: %f, train_acc: %f, test_loss: %f, test_acc: %f \n', ...
        train_time(i), train_loss(i), train_acc(i), test_loss(i), test_acc(i)); 
end
end