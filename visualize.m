% visualize

target = 'mnist';
solvers = {'SGD', 'Momentum', 'Nesterov', 'AdaGrad', 'AdaDelta', 'RMSProp', 'Adam', 'Adamax', 'SAG', 'SVRG'};
markers = {'--^', '--s', '--o', '--*', '--^', '--s', '--o', '--*', '--^', '--s', '--o', '--*'};
curve_point = 20;
for i = 1:length(solvers)
    disp(solvers{i});
    disp(['./result/', target, '_', solvers{i}, '_train_time.mat']);
    train_time = load(['./result/', target, '_', solvers{i}, '_train_time.mat']);
    train_time = cumsum(train_time.train_time);
    test_error = load(['./result/', target, '_', solvers{i}, '_test_acc.mat']);
    test_error = 1 - test_error.test_acc;
    x = train_time(logical(train_time<=1));
    y = test_error(logical(train_time<=1));
    index = 1:round(length(x)/curve_point):length(x);
    plot(x(index), y(index), markers{i});
    hold on;
end
xlabel('Train Time');
ylabel('Test Error');
legend(solvers);