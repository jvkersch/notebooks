function simpson38_vector(f, xinterval, n)
    @assert n % 3 == 0

    a, b = xinterval
    h = (b - a)/n

    I = (f(a) + 
         3*sum(f.(a .+ h*(1:3:(n-1)))) + 
         3*sum(f.(a .+ h*(2:3:(n-1)))) + 
         2*sum(f.(a .+ h*(3:3:(n-1)))) + f(b))*3*h/8
    I
end

function simpson38_loop(f, xinterval, n)
    @assert n % 3 == 0

    a, b = xinterval
    h = (b - a)/n
    
    I = f(a) + f(b)
    for i ∈ 1:(n-1)
        I += (i % 3 == 0 ? 2 : 3)*f(a + h*i)
    end
    I *= 3*h/8

    I
end

sinc(x) = x > 0 ? sin(x)/x : 1

for n ∈ [3, 9, 15, 21]
    println(simpson38_vector(sinc, (0, pi/2), n))
    println(simpson38_loop(sinc, (0, pi/2), n))
end