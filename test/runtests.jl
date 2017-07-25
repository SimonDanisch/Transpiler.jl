using Transpiler
using Base.Test

# TODO, GPUArrays tests most of the functionality of Transpiler.
# Maybe we should just depend on GPUArrays and run the OpenCL tests here?!

function test{T}(a::T, b)
    x = sqrt(sin(a) * b) / T(10.0)
    y = T(33.0)x + cos(b)
    y * T(10.0)
end

function fortest(x)
    acc = x
    for i = 1:5
        if i == 1
            acc += x
        elseif i == 2
            acc -= x
        else
            acc += x * x
        end
    end
    return acc
end

@testset "OpenCL Transpiler" begin
    include("opencl_funcs.jl")
end
# @testset "OpenGL Transpiler" begin
#     include("opengl_funcs.jl")
# end
