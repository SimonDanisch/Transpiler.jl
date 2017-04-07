Pkg.checkout("Sugar", "sd/for")
using Transpiler
using Base.Test

# TODO, GPUArrays tests most of the functionality of Transpiler.
# Maybe we should just depend on GPUArrays and run the OpenCL tests here?!
@testset "CLTranspiler" begin
    include("opencl_funcs.jl")
end
