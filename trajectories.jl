using Random, DifferentialEquations, LinearAlgebra, Turing
using Plots, StatsPlots
#using LinearAlgebra

Random.seed!(1234)
Turing.setadbackend(:forwarddiff)

H = [
    -1 1 0  0 0  0;
     0 0 1 -1 0  0;
     0 0 0  0 1 -1;
]

function ryra!(dx, x, p, t)
    k1, k2, k3, k4, k5, k6 = p
    x1, x2, x3, x4 = x

    dx[1] = -(k1 + k3 + k5) * x1 + k2 * x3 + k4 * x2 + k6 * x4
    dx[2] = k3 * x1 - k4 * x2
    dx[3] = k1 * x1 - k2 * x3
    dx[4] = k5 * x1 - k6 * x4
end

# # CAN REPLACE THIS BY "Array"
to_matrix(vv) = reduce(vcat, transpose.(vv))

function simulate_balance_laws(ks, x0, timepoints)
    k1, k2, k3, k4, k5, k6 = ks
    K = [
        -(k1 + k3 + k5)  k4  k2 k6
                    k3  -k4   0  0
                    k1    0 -k2  0
                    k5    0  0 -k6
    ]

    exact_solution(t) = exp(K * t) * x0
    xs = to_matrix(exact_solution.(timepoints))

    xs
end

function compute_rates(ks, xs)
    k1, k2, k3, k4, k5, k6 = ks
    K = [
       k1  0  0  0;
        0  0 k2  0;
       k3  0  0  0;
        0 k4  0  0;
       k5  0  0  0;
        0  0  0 k6;
    ]
    xs*transpose(K)*transpose(H)
end


add_noise(xs, σ) = xs + σ*randn(rng, size(xs))

function create_experiment(ks, x0, T, n, σ)
    ts = range(0, T, length=n)
    xs = simulate_balance_laws(ks, x0, ts)
    νs = compute_rates(ks, xs)
    νs = add_noise(νs, σ)

    ts_dense = range(0, T, length=100)
    xs_dense = simulate_balance_laws(ks, x0, ts_dense)
    νs_dense = compute_rates(ks, xs_dense)

    ts, νs, ts_dense, νs_dense
end

x0 = [0, 0, 0.963, 0.037]
ks = [28.8, 984.15, 1093.5, 385.9, 1.75, 0.1]
σ = 0.02

ts1, νs1, ts_dense1, νs_dense1 = create_experiment(ks, x0, 10, 11, 0.02)
ts2, νs2, ts_dense2, νs_dense2 = create_experiment(ks, x0,  7, 9, 0.005)

p = plot(ts_dense1, νs_dense1, layout=(3, 1))
scatter!(p, ts1, νs1)

p = plot(ts_dense2, νs_dense2, layout=(3, 1))
scatter!(p, ts2, νs2)

@model function fitryra(data, prob)
    σ ~ InverseGamma(2, 3)
    k1 ~ Uniform(0, 10000)
    k2 ~ Uniform(0, 10000)
    k3 ~ Uniform(0, 10000)
    k4 ~ Uniform(0, 10000)
    k5 ~ Uniform(0, 10000)
    k6 ~ Uniform(0, 10000)
    ks = [k1, k2, k3, k4, k5, k6]

    prob = remake(prob, ks=ks)
    predicted = solve(prob, saveat=1.0)

    for i = 1:length(predicted)
        x1, x2, x3, x4 = predicted[i]
        ν₁ = -k1*x1 + k2*x3
        ν₂ =  k3*x1 - k4*x2
        ν₃ =  k5*x1 - k6*x4
        data[:,i] ~ MvNormal([ν₁, ν₂, ν₃], σ)
    end
end

prob = ODEProblem(ryra!, x0, (0.0,10.0), ks)
model = fitryra(transpose(νs1), prob)

chain = sample(model, NUTS(.65), 1000)
#chain = mapreduce(c -> sample(model, NUTS(.65),1000), chainscat, 1:3)

plot(chain)