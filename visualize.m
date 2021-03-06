% visualize

target = 'mnist';
solvers = {'SGD', 'Momentum', 'Nesterov', 'AdaGrad', 'AdaDelta', 'RMSProp', 'Adam', 'Adamax', 'SAG', 'SVRG'};
markers = {'--^', '--s', '--o', '--*', '--^', '--s', '--o', '--*', '--^', '--s', '--o', '--*'};
curve_point = 20;
for i = 1:length(solvers)
    disp(solvers{i});
    train_time = load(['./result/', target, '_', solvers{i}, '_train_time.mat']);
    train_time = cumsum(train_time.train_time);
    test_acc = load(['./result/', target, '_', solvers{i}, '_test_acc.mat']);
    [max_acc, max_i] = max(test_acc.test_acc);
    disp(['acc: ', num2str(max_acc), '; time: ', num2str(train_time(max_i)), '; TLE: ', num2str(max(train_time))]);
    test_error = 1 - test_acc.test_acc;
    x = train_time(logical(train_time<=240));
    y = test_error(logical(train_time<=240));
    index = 1:round(length(x)/curve_point):length(x);
    plot(x(index), y(index), markers{i});
    hold on;
end
set(gca, 'YScale', 'log');
ylim([0.08, 1]);
xlabel('Train Time (s)');
ylabel('Test Error');
legend(solvers);

figure()
for i = 1:length(solvers)
    train_time = load(['./result/', target, '_', solvers{i}, '_train_time.mat']);
    train_time = cumsum(train_time.train_time);
    test_acc = load(['./result/', target, '_', solvers{i}, '_test_acc.mat']);
    test_error = 1 - test_acc.test_acc;
    index = 1:round(length(test_error)/curve_point):length(test_error);
    plot(test_error(index), markers{i});
    hold on;
end
set(gca, 'YScale', 'log');
xlim([1, 20]);
ylim([0.08, 1]);
xlabel('Iterations (10^3)');
ylabel('Test Error');
legend(solvers);
