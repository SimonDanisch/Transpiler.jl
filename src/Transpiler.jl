__precompile__(true)
module Transpiler

using StaticArrays, Compat
using Sugar, DataStructures
using OpenCL
#using GLAbstraction, ModernGL


include("clike/shared.jl")
#include("clike/opengl/compilation.jl")
include("clike/opencl/compilation.jl")


function empty_caches!()
    empty_replace_cache!()
    #gl_empty_compile_cache!()
    cl_empty_compile_cache!()
    return
end

end # module
