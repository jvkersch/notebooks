using OrdinaryDiffEqTsit5
using StaticArrays
using ForwardDiff
using LinearAlgebra

"""
    vanderpol(x; T=2π, u_0=1.0, p_0=0.0)

Integrate the van der Pol equation ü - μ(1-u²)u̇ + ω²u = 0 from t=0 to t=T
and return u(T). Parameters are in normalized [-1,1]² coordinates:
ω = 1.0 + 0.2*x[1], μ = 2.5 + 2.5*x[2].
"""
function vanderpol(x; T=2π, u_0=1.0, p_0=0.0)
    ω = 1.0 + 0.2 * x[1]
    μ = 2.5 + 2.5 * x[2]

    function rhs(u, params, t)
        ω, μ = params
        SA[u[2], μ * (1 - u[1]^2) * u[2] - ω^2 * u[1]]
    end

    prob = ODEProblem(rhs, SA[u_0, p_0], (0.0, T), SA[ω, μ])
    sol = solve(prob, Tsit5(); reltol=1e-10, abstol=1e-10)
    sol[1, end]
end

"""
    vanderpol_gradient(x; T=2π, u_0=1.0, p_0=0.0)

Compute [∂f/∂x₁, ∂f/∂x₂] via forward-mode autodiff through the ODE solve.
"""
function vanderpol_gradient(x; T=2π, u_0=1.0, p_0=0.0)
    ForwardDiff.gradient(p -> vanderpol(p; T, u_0, p_0), SA[x[1], x[2]])
end

"""
    active_subspace_matrix(samples; T=2π, u_0=1.0, p_0=0.0)

Monte Carlo approximation of the active subspace matrix C = E[∇f ∇fᵀ].
`samples` is an (n, 2) array where each row is a point in [-1,1]².
"""
function active_subspace_matrix(samples; T=2π, u_0=1.0, p_0=0.0)
    n = size(samples, 1)
    C = zeros(2, 2)
    for i in 1:n
        g = vanderpol_gradient(samples[i, :]; T, u_0, p_0)
        C .+= g * g'
    end
    C ./= n
end

"""
    active_subspace(samples; T=2π, u_0=1.0, p_0=0.0)

Compute the active subspace decomposition from `(n, 2)` samples in [-1,1]².
Returns `(eigenvalues, eigenvectors)` sorted by decreasing eigenvalue.
The first eigenvector is the active subspace direction.
"""
function active_subspace(samples; T=2π, u_0=1.0, p_0=0.0)
    C = active_subspace_matrix(samples; T, u_0, p_0)
    F = eigen(Symmetric(C), sortby=x -> -x)
    F.values, F.vectors
end

"""
    fit_surrogate(X, f, w; degree=5)

Fit a univariate polynomial surrogate in the active subspace variable y = wᵀx.
Returns a function that maps a 2D point x to the surrogate prediction.
"""
function fit_surrogate(X, f, w; degree=5)
    y = X * w
    # Vandermonde matrix for polynomial fit
    V = hcat([y .^ k for k in 0:degree]...)
    coeffs = V \ f
    function surrogate(x)
        yi = dot(w, x)
        sum(coeffs[k+1] * yi^k for k in 0:degree)
    end
end
