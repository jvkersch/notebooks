using Plots

bernstein(n, ν, x) = binomial(n, ν)*x^ν*(1-x)^(n-ν)

approx(f, n) = x -> sum([
    f(ν/n)*bernstein(n, ν, x) for ν = 0:n
])

n = 41
f(x) = sin(π*x)
approx_f = approx(f, n)

xs = range(0, 1, step=0.01)
ys_approx = approx_f.(xs)
ys_exact = f.(xs)

label = ["Exact" "Approx, n = $(n)"]
plot(xs, [ys_exact ys_approx], label = label)