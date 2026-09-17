include("vanderpol.jl")
using CairoMakie, Random, Statistics

# Training data
Random.seed!(12345)
X_train = 2 * rand(1000, 2) .- 1
f_train = [vanderpol(SA[X_train[i, 1], X_train[i, 2]]) for i in 1:size(X_train, 1)]

# Compute active subspace direction (sign-matched to regression)
λ, W = active_subspace(X_train)
w_reg = (hcat(ones(size(X_train, 1)), X_train) \ f_train)[2:3]
w_as = sign(dot(W[:, 1], w_reg)) * W[:, 1]

# Fit surrogate
surrogate = fit_surrogate(X_train, f_train, w_as; degree=5)

# Test data
Random.seed!(99999)
X_test = 2 * rand(500, 2) .- 1
f_test = [vanderpol(SA[X_test[i, 1], X_test[i, 2]]) for i in 1:size(X_test, 1)]
f_surr = [surrogate(X_test[i, :]) for i in 1:size(X_test, 1)]

rmse = sqrt(mean((f_test .- f_surr) .^ 2))
println("RMSE on test set: ", rmse)

# Plot
fig = Figure(size=(1100, 500))

# Left: sufficient summary with surrogate curve
y_train = X_train * w_as
y_curve = range(extrema(y_train)..., length=200)
f_curve = [surrogate(y * w_as) for y in y_curve]

ax1 = Axis(fig[1, 1], xlabel="w₁ᵀx", ylabel="f(x)", title="Surrogate fit (degree 5)")
scatter!(ax1, y_train, f_train, markersize=4, color=(:steelblue, 0.3), label="Training data")
lines!(ax1, collect(y_curve), f_curve, color=:red, linewidth=2, label="Polynomial surrogate")
axislegend(ax1, position=:lb)

# Right: true vs predicted
ax2 = Axis(fig[1, 2], xlabel="True f(x)", ylabel="Surrogate f̂(x)",
    title="Test set (RMSE = $(round(rmse, digits=3)))", aspect=1)
scatter!(ax2, f_test, f_surr, markersize=4, color=(:steelblue, 0.5))
lims = (min(minimum(f_test), minimum(f_surr)) - 0.1, max(maximum(f_test), maximum(f_surr)) + 0.1)
lines!(ax2, [lims...], [lims...], color=:red, linewidth=1.5, linestyle=:dash, label="y = x")
xlims!(ax2, lims)
ylims!(ax2, lims)
axislegend(ax2, position=:rb)

save("surrogate_plot.png", fig, px_per_unit=2)
