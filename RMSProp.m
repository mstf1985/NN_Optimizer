function [train_loss, train_acc, test_loss, test_acc, train_time] ...
    = RMSProp(gradFunc, lossFunc, accFunc, x_train, y_train, x_test, y_test, opts)
% RMSProp solver
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
if ~isfield(opts, 'rho')
    rho = 0.9;
else
    rho = opts.rho;
end

[~, m] = size(x_train);
G = zeros(m, 1);
w = randn(m, 1);
train_loss = zeros(max_iter, 1);
train_acc = zeros(max_iter, 1);
test_loss = zeros(max_iter, 1);
test_acc = zeros(max_iter, 1);
train_time = zeros(max_iter, 1);
for i = 1:max_iter
    tic;
    fprintf('iter: %d/%d\n', i, max_iter);
    g = gradFunc(y_train, w, x_train, lamb);
    G = rho*G + (1-rho)*g.^2;
    w = w - lr .* g ./ sqrt(G + eps);
    train_loss(i) = lossFunc(y_train, w, x_train, lamb);
    train_acc(i) = accFunc(y_train, w, x_train);
    test_loss(i) = lossFunc(y_test, w, x_test, 0); % test loss doesn't include regulation
    test_acc(i) = accFunc(y_test, w, x_test);
    train_time(i) = toc;
    fprintf('CPU time: %f, train loss: %f, train_acc: %f, test_loss: %f, test_acc: %f \n', ...
        train_time(i), train_loss(i), train_acc(i), test_loss(i), test_acc(i)); 
end
end