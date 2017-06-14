function [proba, loss] = Softloss(y, w, x, lamb)
% proba, loss and gradient
% - x: n_samples, n_features
% - w: n_features, n_labels
% - y: n_samples, n_labels

n_samples = size(x, 1);
ExW = exp(x*w);
proba = bsxfun(@rdivide, ExW, sum(ExW, 2));
loss  = -sum(sum(log(proba).*y))/n_samples + lamb/2 * sum(sum(w.^2));
% grad  = -x'*(y - proba)/ n_samples + lamb*w;

end