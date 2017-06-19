function [train_loss, train_acc, test_loss, test_acc, train_time] ...
    = Adamax(x_train, y_train, x_test, y_test, opts)
% Adamax solver
if ~isfield(opts, 'lr')
    lr = 0.002;
else
    lr = opts.lr;
end
if ~isfield(opts, 'max_iter')
    max_iter = 300;
else
    max_iter = opts.max_iter;
end
if ~isfield(opts, 'lamb')
    lamb = 0;
else
    lamb = opts.lamb;
end
if ~isfield(opts, 'eps')
    eps = 1e-8;
else
    eps = opts.eps;
end
if ~isfield(opts, 'rho1')
    rho1 = 0.9;
else
    rho1 = opts.rho1;
end
if ~isfield(opts, 'rho2')
    rho2 = 0.999;
else
    rho2 = opts.rho2;
end
if ~isfield(opts, 'batch_size')
    batch_size = 100;
else
    batch_size = opts.batch_size;
end

rng('default'); % random seed
[n_samples, n_features] = size(x_train);
n_batch = round(n_samples / batch_size);
[~, n_labels] = size(y_train);
w = randn(n_features, n_labels);
r = zeros(n_features, n_labels);
s = zeros(n_features, n_labels);
train_loss = zeros(max_iter, 1);
train_acc = zeros(max_iter, 1);
test_loss = zeros(max_iter, 1);
test_acc = zeros(max_iter, 1);
train_time = zeros(max_iter, 1);

for i = 1:max_iter
    tic;
    fprintf('iter: %d/%d\n', i, max_iter);
    si = randsample(1:n_batch, 1);
    batch = ((si - 1)*batch_size + 1):si*batch_size;
    g = Softgrad(y_train(batch, :, :), w, x_train(batch, :, :), lamb);
    s = rho1 * s + (1 - rho1) * g;
    r = max(rho2 * r, abs(g));
    s_hat = s / (1 - rho1^i);
    w = w - lr .* (s_hat ./(r + eps));
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