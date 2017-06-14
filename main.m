% main

%% parameters
data_path = './data/';
result_path = './result/';
target = 'mnist';
solver = 'AdaDelta';
opts.lr = 0.01;
opts.max_iter = 200;
opts.lamb = 0.001;
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

%% train and test 
onehot = @(y) bsxfun(@eq, y(:), 1:max(y));
n_train = round(0.75*length(data));
x_train = data(1:n_train, 1:end-1);
y_train = data(1:n_train, end)+1;
y_train = onehot(y_train);
x_test = data(n_train+1:end, 1:end-1);
y_test = data(n_train+1:end, end)+1;
y_test = onehot(y_test);

%% train
Solver = str2func(solver);
[train_loss, train_acc, test_loss, test_acc, train_time] = Solver(x_train, y_train, x_test, y_test, opts);

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