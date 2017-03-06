module CLTranspiler

using Sugar, OpenCL
using OpenCL: cl
import Sugar: ssavalue_name, ASTIO, get_slottypename, get_type, LazyMethod
import Sugar: getsource!, dependencies!, istype, isfunction, getfuncargs, isintrinsic

const GLMethod = LazyMethod{:GL}

include("intrinsics.jl")
include("printing.jl")
include("rewriting.jl")

immutable ComputeProgram{Args <: Tuple}
    program::cl.Kernel
end

_to_glsl_types(::Type{Int32}) = Int32
_to_glsl_types(::Type{Int64}) = Int32
_to_glsl_types(::Type{Float32}) = Float32
_to_glsl_types(::Type{Float64}) = Float32
_to_glsl_types{T}(arg::T) = _to_glsl_types(T)
_to_glsl_types{T}(::Type{T}) = T

function _to_glsl_types{T <: cl.Buffer}(arg::T)
    return cli.CLArray{eltype(arg), ndims(arg)}
end
function to_glsl_types(args::Union{Vector, Tuple})
    map(_to_glsl_types, args)
end

const compiled_functions = Dict{Any, ComputeProgram}()

function ComputeProgram{T}(f::Function, args::T, ctx; local_size = (16, 16, 1))
    gltypes = to_glsl_types(args)
    get!(compiled_functions, (f, gltypes)) do # TODO make this faster
        decl = GLMethod((f, gltypes))
        funcsource = getsource!(decl)
        # add compute program dependant infos
        io = CLIO(IOBuffer(), decl)
        deps = reverse(collect(dependencies!(decl, true)))
        types = filter(istype, deps)
        funcs = filter(isfunction, deps)
        println(io, "// dependant type declarations")
        for typ in types
            if !isintrinsic(typ)
                println(io, getsource!(typ))
            end
        end
        println(io, "// dependant function declarations")
        for func in funcs
            if !isintrinsic(func)
                println(io, getsource!(func))
            end
        end

        println(io, "// Main inner function")
        print(io, "__kernel ") # mark as kernel function
        println(io, funcsource)
        kernelsource = String(take!(io.io))
        println(kernelsource)
        p = cl.build!(cl.Program(ctx, source = kernelsource))
        fname = string(glsl_name(Sugar.getfunction(decl)))
        k = cl.Kernel(p, fname)
        ComputeProgram{T}(k)
    end::ComputeProgram{T}
end




end
