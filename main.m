% train
% min 1/n*sum(f_i(w)+lamd*||w||_1 [2]
% f_i(w) = log(1+exp(-y^i*w^T*x^i))

%% parameters
data_path = '../data/';
result_path = '../result/';
target = 'covtype';
solver = 'RMSProp';
opts.lr = 0.01;
opts.max_iter = 500;
opts.lamb = 10;
prefix = sprintf('%s_%s_lr%s_lamb%s_iter%d', ...
    target, solver, num2str(opts.lr), num2str(opts.lamb), opts.max_iter);

%% load and shuffle data
rng('default'); % random seed
if ~exist(target, 'var')
    disp(['load ', target, ' data']);
    load([data_path, target ,'.mat']);
    data = eval(target);
    data = data(randperm(length(data)), :);
end

%% L2 norm
% loss = @(y, w, x, lamb) sum(log(1+exp(-y.*(x*w))))/numel(y) + 0.5*lamb*norm(w, 2);
% grad = @(y, w, x, lamb) -x.'*(y./(1+exp(y.*(x*w))))/numel(y) + lamb*w;
%% L1 norm
% grad = @(y, w, x, lamb) -x.'*(y./(1+exp(y.*(x*w))))/numel(y) + lamb*sign(w);
loss = @(y, w, x, lamb) sum(log(1+exp(-y.*(x*w))))/numel(y) + lamb*norm(w, 1);
grad = @(y, w, x, lamb) Grad(y, w, x, lamb);
accuracy = @(y, w, x) sum(sign(x*w)==y)/numel(y);

%% train and test 
n_train = round(0.75*length(data));
x_train = data(1:n_train, 1:end-1);
y_train = data(1:n_train, end);
x_test = data(n_train+1:end, 1:end-1);
y_test = data(n_train+1:end, end);

%% train
Solver = str2func(solver);
[train_loss, train_acc, test_loss, test_acc, train_time] = Solver(grad, loss, accuracy, x_train, y_train, x_test, y_test, opts);

%% plot
figure();
plot(train_loss);
hold on;
plot(test_loss);
legend('train loss', 'test loss');
saveas(gcf, [result_path, prefix, '_loss.png']);
figure();
plot(train_acc);
hold on;
plot(test_acc);
legend('train acc', 'test acc', 'Location','SouthEast');
saveas(gcf, [result_path, prefix, '_acc.png']);

%% save result
save([result_path, prefix, '_train_loss.mat'], 'train_loss');
save([result_path, prefix, '_train_acc.mat'], 'train_acc');
save([result_path, prefix, '_test_loss.mat'], 'test_loss');
save([result_path, prefix, '_test_acc.mat'], 'test_acc');
save([result_path, prefix, '_train_time.mat'], 'train_time');