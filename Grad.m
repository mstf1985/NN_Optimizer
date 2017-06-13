function g = Grad(y, w, x, lamb)
dL = -x'*(y./(1+exp(y.*(x*w))))/numel(y);
g = dL + lamb*sign(w);
g(w==0 & dL<-lamb) = dL(w==0 & dL<-lamb) + lamb;
g(w==0 & dL>lamb) = dL(w==0 & dL>lamb) - lamb;
g(w==0 & -lamb<=dL<=lamb) = 0;
end