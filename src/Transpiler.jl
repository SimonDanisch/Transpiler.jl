module Transpiler

using StaticArrays
using Compat

include("clike/shared.jl")
#include("clike/opengl/compilation.jl")
include("clike/opencl/compilation.jl")


function empty_caches!()
    empty_replace_cache!()
    CLTranspiler.empty_compile_cache!()
    return
end

end # module
