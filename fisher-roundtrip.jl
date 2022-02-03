using Distributions
using ForwardDiff
using LinearAlgebra
using Optim

dist = MvNormal([1, 3], [2 0; 0 3])

mle(dist) = Optim.minimizer(optimize(x -> -logpdf(dist, x), zeros(length(dist))))
fisher_info(dist) = -ForwardDiff.hessian(x -> logpdf(dist, x), mle(dist))
laplace_approx_sd(dist) = diag(inv(fisher_info(dist))).^0.5

println(laplace_approx_sd(dist).^2)