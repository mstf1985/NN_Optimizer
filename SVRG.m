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
if ~isfield(opts, 'lamb')
    lamb = 1.;
else
    lamb = opts.lamb;
end
if ~isfield(opts, 'batch_size')
    batch_size = 100;
else
    batch_size = opts.batch_size;
end
if ~isfield(opts, 'freq')
    freq = 10;
else
    freq = opts.freq;
end

rng('default'); % random seed
[n_samples, n_features] = size(x_train);
n_batch = round(n_samples / batch_size);
[~, n_labels] = size(y_train);
w = randn(n_features, n_labels);
train_loss = zeros(max_iter, 1);
train_acc = zeros(max_iter, 1);
test_loss = zeros(max_iter, 1);
test_acc = zeros(max_iter, 1);
train_time = zeros(max_iter, 1);
for i = 1:max_iter
    tic;
    fprintf('iter: %d/%d\n', i, max_iter);
    s = randsample(1:n_batch, 1);
    batch = ((s-1)*n_batch+1):s*n_batch;
    g = Softgrad(y_train(batch, :, :), w, x_train(batch, :, :), lamb);
    d = w;
    for k = 1:freq
        s = randsample(batch, 1);
        g_d = Softgrad(y_train(s, :, :), d, x_train(s, :, :), lamb);
        g_w = Softgrad(y_train(s, :, :), w, x_train(s, :, :), lamb);
        v = g_d - g_w + g;
        d = d - lr .*  v;
    end
    w = d;
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