function [pred, loss, grad] = Softmax(y, w, x, lamb)
% prediction, loss and gradient
% - x: n_samples, n_features
% - w: n_features, n_labels
% - y: n_samples, n_labels

n_samples = size(x, 1);
ExM = exp(x*w);
pred = bsxfun(@rdivide, ExM', sum(ExM'))';
loss = -sum(sum(log(pred).*y))/n_samples + lamb/2 * sum(sum(w.^2));
grad = -x'*(y - pred)/ n_samples + lamb*w;

end