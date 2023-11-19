using Metal
using KernelAbstractions
import LinearAlgebra: qr!

function my_qr(mtl_arr::MtlArray)
    intermediate = unsafe_wrap(Array{Float64}, mtl_arr, size(mtl_arr))
    qr!(intermediate)
end

backend_private = MetalBackend()
A = KernelAbstractions.allocate(backend_private, Float32, 3, 3)
println(Metal.storagemode(A))

backend_shared = MetalBackend(storage=Shared)
B = KernelAbstractions.allocate(backend_shared, Float32, 3, 3)
println(Metal.storagemode(B))

# B_cpu = unsafe_wrap(Array{Float32}, B, size(B))
# println(qr!(B_cpu))
println(my_qr(B))