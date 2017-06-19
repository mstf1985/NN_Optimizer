function [train_loss, train_acc, test_loss, test_acc, train_time] ...
    = SAG(x_train, y_train, x_test, y_test, opts)
% SAG solver
if ~isfield(opts, 'lr')
    lr = 1e-2;
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
if ~isfield(opts, 'batch_size')
    batch_size = 100;
else
    batch_size = opts.batch_size;
end

rng('default'); % random seed
[n_samples, n_features] = size(x_train);
n_batch = round(n_samples / batch_size);
[~, n_labels] = size(y_train);
d = zeros(n_features, n_labels);
D = zeros(n_batch, n_features, n_labels);
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
    batch = ((s - 1)*batch_size + 1):s*batch_size;
    g = Softgrad(y_train(batch, :, :), w, x_train(batch, :, :), lamb);
    d = d - squeeze(D(s, :, :)) + g;
    D(s, :, :) = g;
    w = w - lr .* d;
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