__precompile__(true)
module Transpiler

using StaticArrays, Compat
using Sugar, DataStructures
#using GLAbstraction, ModernGL


include("clike/shared.jl")
#include("clike/opengl/compilation.jl")
include("clike/opencl/printing.jl")


function empty_caches!()
    empty_replace_cache!()
    return
end

end # module
