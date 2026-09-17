include("vanderpol.jl")
using Random

Random.seed!(12345)
samples = 2 * rand(1000, 2) .- 1  # uniform on [-1, 1]²

λ, W = active_subspace(samples)

println("Active subspace matrix C:")
display(active_subspace_matrix(samples))
println("\nEigenvalues: ", λ)
println("\nActive subspace direction: ", W[:, 1])
println("Inactive direction:        ", W[:, 2])
