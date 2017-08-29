__precompile__(true)
module Transpiler

using StaticArrays, Compat
using Sugar, DataStructures


include("clike/shared.jl")
include("clike/opencl/printing.jl")


function empty_caches!()
    empty_replace_cache!()
    empty_hash_dict!()
    return
end

end # module
