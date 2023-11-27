using Plots
using SpecialFunctions
using LaTeXStrings

inverse_chi_squared(x, ν) = 2^(-ν/2)/gamma(ν/2)*x^(-ν/2-1)*exp(-1/(2*x))

x = range(0, 1, length = 100)
p = plot(title=L"\textrm{Inv.}\chi^2(\nu)")
for ν in [1, 2, 3, 4, 5]
    y = inverse_chi_squared.(x, ν)
    plot!(p, x, y, label="ν = $ν", linewidth=3)
end
display(p)
savefig(p, "inv_chi_square.png")