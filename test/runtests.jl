#TODO register these packages!
for pkg in ("Matcha", "Sugar")
    installed = try
        Pkg.installed(pkg) != nothing
    catch e
        false
    end
    installed || Pkg.clone("https://github.com/SimonDanisch/$(pkg).jl.git")
end

using Transpiler
using Base.Test

# TODO, GPUArrays tests most of the functionality of Transpiler.
# Maybe we should just depend on GPUArrays and run the OpenCL tests here?!
@testset "CLTranspiler" begin
    include("opencl_funcs.jl")
end
