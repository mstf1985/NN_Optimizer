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

rng('default'); % random seed
[n_samples, n_features] = size(x_train);
n_batch = round(n_samples / batch_size);
[~, n_labels] = size(y_train);
w_hat = randn(n_features, n_labels);
freq = 2 * n_batch;
train_loss = zeros(max_iter, 1);
train_acc = zeros(max_iter, 1);
test_loss = zeros(max_iter, 1);
test_acc = zeros(max_iter, 1);
train_time = zeros(max_iter, 1);
i = 0;
% initialize with SGD
s = randsample(1:n_batch, 1);
batch = ((s - 1)*batch_size + 1):s*batch_size;
g = Softgrad(y_train(batch, :, :), w_hat, x_train(batch, :, :), lamb);
w_hat = w_hat - lr .* g;
while 1
    w = w_hat;
    g = Softgrad(y_train, w_hat, x_train, lamb); % full gradient
    m = zeros(n_features, n_labels);
    for k = 1:freq
        i = i + 1;
        if i > max_iter
            break
        end
        tic;
        fprintf('iter: %d/%d\n', i, max_iter);
        s = randsample(1:n_batch, 1);
        batch = ((s - 1)*batch_size + 1):s*batch_size;
        g_w = Softgrad(y_train(batch, :, :), w, x_train(batch, :, :), lamb);
        g_w_hat = Softgrad(y_train(batch, :, :), w_hat, x_train(batch, :, :), lamb);
        w = w - lr .*  (g_w - g_w_hat + g);
        m = m + w;
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
    if i > max_iter
        break
    end
    w_hat = m / freq;
end
end