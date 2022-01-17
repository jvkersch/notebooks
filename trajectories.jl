using Random, DifferentialEquations, LinearAlgebra, Turing
using Plots, StatsPlots

Random.seed!(1234)
Turing.setadbackend(:forwarddiff)

function ryra!(dx, x, ks, t)
    k1, k2, k3, k4, k5, k6 = ks
    x1, x2, x3, x4 = x

    dx[1] = -(k1 + k3 + k5) * x1 + k2 * x3 + k4 * x2 + k6 * x4
    dx[2] = k3 * x1 - k4 * x2
    dx[3] = k1 * x1 - k2 * x3
    dx[4] = k5 * x1 - k6 * x4
end

function simulate_balance_laws(ks, x0, timepoints)
    prob = ODEProblem(ryra!, x0, (timepoints[1], timepoints[end]), ks)
    sol = solve(prob, saveat=timepoints)
    xs = sol.u

    xs
end

function compute_rates(ks, x)
    k1, k2, k3, k4, k5, k6 = ks
    x1, x2, x3, x4 = x

    return [
        k2*x3 - k1*x1,
        k3*x1 - k4*x2,
        k5*x1 - k6*x4
    ]
end

add_noise(xs, σ) = xs + σ*randn(size(xs))

function create_experiment(ks, x0, T, n, σ)
    ts = range(0, T, length=n)
    xs = simulate_balance_laws(ks, x0, ts)
    νs = compute_rates.(Ref(ks), xs)

    νs = to_matrix(νs)
    νs = add_noise(νs, σ)

    ts_dense = range(0, T, length=100)
    xs_dense = simulate_balance_laws(ks, x0, ts_dense)
    νs_dense = compute_rates.(Ref(ks), xs_dense)
    νs_dense = to_matrix(νs_dense)

    ts, νs, ts_dense, νs_dense
end

# TODO: remove the transpose from the evocation
@model function fitryra(ts1, νs1, ts2, νs2, prob)
    # Note: data is a 3 x k array of the observed reaction rates

    σ1² ~ InverseGamma(2, 3)
    σ2² ~ InverseGamma(2, 3)

    k1 ~ Uniform(0, 10000)
    k2 ~ Uniform(0, 10000)
    k3 ~ Uniform(0, 10000)
    k4 ~ Uniform(0, 10000)
    k5 ~ Uniform(0, 10000)
    k6 ~ Uniform(0, 10000)
    ks = [k1, k2, k3, k4, k5, k6]
    prob = remake(prob, ks=ks)

    # Forward solve dataset 1
    predicted = solve(prob, saveat=ts1)

    for i = 1:length(predicted)
        x1, x2, x3, x4 = predicted[i]
        ν₁ = -k1*x1 + k2*x3
        ν₂ =  k3*x1 - k4*x2
        ν₃ =  k5*x1 - k6*x4
        νs1[:,i] ~ MvNormal([ν₁, ν₂, ν₃], sqrt(σ1²))
    end

    # Forward solve dataset 2
    predicted = solve(prob, saveat=ts2)

    for i = 1:length(predicted)
        x1, x2, x3, x4 = predicted[i]
        ν₁ = -k1*x1 + k2*x3
        ν₂ =  k3*x1 - k4*x2
        ν₃ =  k5*x1 - k6*x4
        νs2[:,i] ~ MvNormal([ν₁, ν₂, ν₃], sqrt(σ2²))
    end
end

to_matrix(vv) = reduce(vcat, transpose.(vv))

x0 = [0, 0, 0.963, 0.037]
ks = (28.8, 984.15, 1093.5, 385.9, 1.75, 0.1)
σ = 0.02

ts1, νs1, ts_dense1, νs_dense1 = create_experiment(ks, x0, 10, 11, 0.02)
ts2, νs2, ts_dense2, νs_dense2 = create_experiment(ks, x0,  7, 9, 0.005)

exact_label = ["exact" "exact" "exact"]
sample_label = ["samples" "samples" "samples"]

p = plot(ts_dense1, νs_dense1, layout=(3, 1), label=exact_label)
scatter!(p, ts1, νs1, label=sample_label)
display(p)
savefig("samples1.png")

p = plot(ts_dense2, νs_dense2, layout=(3, 1), label=exact_label)
scatter!(p, ts2, νs2, label=sample_label)
display(p)
savefig("samples2.png")

prob = ODEProblem(ryra!, x0, (0.0,10.0), ks)
model = fitryra(ts1, transpose(νs1), ts2, transpose(νs2), prob)

chain = sample(model, NUTS(0.65), MCMCThreads(), 1000, 4)
#chain = sample(model, NUTS(.65), 500)
#chain = mapreduce(c -> sample(model, NUTS(.65),1000), chainscat, 1:4)

p = plot(chain)
display(p)
# savefig(p, "predictions.png")