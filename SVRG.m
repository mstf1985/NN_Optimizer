function [train_loss, train_acc, test_loss, test_acc, train_time] ...
    = SVRG(x_train, y_train, x_test, y_test, opts)
% SVRG solver
if ~isfield(opts, 'lr')
    lr = 1e-3;
else
    lr = opts.lr;
end
if ~isfield(opts, 'max_iter')
    max_iter = 300;
else
    max_iter = opts.max_iter;
end
if ~isfield(opts, 'eps')
    eps = 1e-8;
else
    eps = opts.eps;
end
if ~isfield(opts, 'lamb')
    lamb = 1.;
else
    lamb = opts.lamb;
end
if ~isfield(opts, 'freq')
    freq = 100;
else
    freq = opts.freq;
end

rng('default'); % random seed
[n_samples, n_features] = size(x_train);
[~, n_labels] = size(y_train);
d = zeros(n_features, n_labels);
W = zeros(freq, n_features, n_labels);
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
    s = randsample(1:n_samples, freq);
    W(1, :, :) = w;
    for k = 1:freq-1
        v = Softgrad(y_train(s(k), :), squeeze(W(k, : ,:)), x_train(s(k), :), lamb) - ...
            Softgrad(y_train(s(k), :), w, x_train(s(k), :), lamb) + g;
        W(k+1, :, :) = W(k, :, :) - reshape(lr .*  v, [1, size(v)]);
    end
    w = squeeze(mean(W, 1));
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