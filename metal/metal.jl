using Metal
using LinearAlgebra

arr_mtl = Metal.ones(Float32, (16,16); storage=Shared)
arr_cpu = unsafe_wrap(Array{Float32}, arr_mtl, size(arr_mtl))

qr!(arr_cpu)