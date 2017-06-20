function [train_loss, train_acc, test_loss, test_acc, train_time] ...
    = SGD_Batch(x_train, y_train, x_test, y_test, opts)
% SGD solver
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
    lamb = 1.;
else
    lamb = opts.lamb;
end
if ~isfield(opts, 'batch_size')
    batch_size = 100;
else
    batch_size = opts.batch_size;
end
if ~isfield(opts, 'period')
    period = 100;
else
    period = opts.period;
end

rng('default'); % random seed
[n_samples, n_features] = size(x_train);
[~, n_labels] = size(y_train);
w = randn(n_features, n_labels);
n_period = round(max_iter / period);
train_loss = zeros(n_period, 1);
train_acc = zeros(n_period, 1);
test_loss = zeros(n_period, 1);
test_acc = zeros(n_period, 1);
train_time = zeros(n_period, 1);
time = 0;
for i = 1:max_iter
    tic;
    s = randsample(1:n_samples, batch_size);
    g = Softgrad(y_train(s, :, :), w, x_train(s, :, :), lamb);
    w = w - lr .* g;
    time = time + toc;
    if mod(i, period) == 0
        fprintf('iter: %d/%d\n', i, max_iter);
        t = round(i / period);
        train_time(t) = time;
        time = 0;
        % train eval
        [acc, loss, ~] = Softloss(y_train, w, x_train, lamb);
        train_loss(t) = loss;
        train_acc(t) = acc;
        % test eval
        [acc, loss, ~] = Softloss(y_test, w, x_test, lamb);
        test_loss(t) = loss;
        test_acc(t) = acc;
        fprintf('CPU time: %f, train loss: %f, train_acc: %f, test_loss: %f, test_acc: %f \n', ...
            train_time(t), train_loss(t), train_acc(t), test_loss(t), test_acc(t)); 
    end
end
end