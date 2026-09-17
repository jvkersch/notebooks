include("vanderpol.jl")
using CairoMakie, Random

x1s = range(-1, 1, length=50)
x2s = range(-1, 1, length=50)

Z = [vanderpol(SA[x1, x2]) for x2 in x2s, x1 in x1s]

# Compute active subspace direction
Random.seed!(12345)
samples = 2 * rand(1000, 2) .- 1
λ, W = active_subspace(samples)
w = W[:, 1]

fig = Figure(size=(700, 500))
ax = Axis(fig[1, 1], xlabel="x₁", ylabel="x₂", title="Van der Pol: u(T) at T = 2π", aspect=1)
ct = contourf!(ax, x1s, x2s, Z, levels=20)
contour!(ax, x1s, x2s, Z, levels=20, color=:black, linewidth=0.5, labels=true, labelsize=12, labelfont=:bold)

# Draw active subspace direction as an arrow through the origin
arrows!(ax, [0.0], [0.0], [w[1]], [w[2]], color=:red, linewidth=3, arrowsize=15)
arrows!(ax, [0.0], [0.0], [-w[1]], [-w[2]], color=:red, linewidth=3, arrowsize=15)

Colorbar(fig[1, 2], ct, label="u(T)")
save("contour_plot.png", fig, px_per_unit=2)
