include("vanderpol.jl")
using CairoMakie, Random

Random.seed!(12345)
X = 2 * rand(1000, 2) .- 1
f = [vanderpol(SA[X[i, 1], X[i, 2]]) for i in 1:size(X, 1)]

# Fit linear model: f ≈ β₀ + β₁*x₁ + β₂*x₂
A = hcat(ones(size(X, 1)), X)
β = A \ f
w_reg = β[2:3]
w_reg = w_reg / norm(w_reg)

# Gradient-based active subspace direction
λ, W = active_subspace(X)
# Eigenvectors are defined up to sign; flip to match the regression direction
w_as = sign(dot(W[:, 1], w_reg)) * W[:, 1]

println("Regression direction:      ", w_reg)
println("Active subspace direction: ", w_as)

fig = Figure(size=(1100, 500))

ax1 = Axis(fig[1, 1], xlabel="wᵀx", ylabel="f(x)", title="Linear regression direction")
scatter!(ax1, X * w_reg, f, markersize=4, color=(:steelblue, 0.5))

ax2 = Axis(fig[1, 2], xlabel="w₁ᵀx", ylabel="f(x)", title="Active subspace direction")
scatter!(ax2, X * w_as, f, markersize=4, color=(:steelblue, 0.5))

save("sufficient_summary_plot.png", fig, px_per_unit=2)
