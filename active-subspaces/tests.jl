include("vanderpol.jl")
using LinearAlgebra

println("=== Test 1: Gradient vs finite differences ===")
ω, μ = 1.0, 2.0
h = 1e-6
g_ad = vanderpol_gradient(ω, μ)
g_fd = [
    (vanderpol(ω + h, μ) - vanderpol(ω - h, μ)) / (2h),
    (vanderpol(ω, μ + h) - vanderpol(ω, μ - h)) / (2h),
]
println("AD gradient:  ", g_ad)
println("FD gradient:  ", g_fd)
println("Abs error:    ", abs.(g_ad .- g_fd))
println()

println("=== Test 2: μ=0 reduces to harmonic oscillator ===")
# u'' + ω²u = 0, u(0)=1, u'(0)=0  =>  u(t) = cos(ωt)
# u(T) = cos(ωT), with T=2π: u(2π) = cos(2πω)
# ∂u/∂ω = -2π sin(2πω),  ∂u/∂μ = 0
for ω_test in [0.8, 1.0, 1.2]
    u_num = vanderpol(ω_test, 0.0)
    u_exact = cos(2π * ω_test)
    g_num = vanderpol_gradient(ω_test, 0.0)
    g_exact_omega = -2π * sin(2π * ω_test)
    println("ω=$ω_test: u_num=$u_num, u_exact=$u_exact, err=$(abs(u_num - u_exact))")
    println("  ∂u/∂ω: num=$(g_num[1]), exact=$g_exact_omega, err=$(abs(g_num[1] - g_exact_omega))")
    println("  ∂u/∂μ: num=$(g_num[2]) (should be ≈0)")
end
println()

println("=== Test 3: Active subspace matrix symmetry and positive semi-definiteness ===")
using Random
Random.seed!(99)
samples = hcat(0.8 .+ 0.4 * rand(200), 5.0 * rand(200))
C = active_subspace_matrix(samples)
println("C symmetric? max|C - C'| = ", maximum(abs.(C - C')))
evals = eigvals(C)
println("Eigenvalues: ", evals, " (all ≥ 0? ", all(evals .>= -1e-14), ")")
