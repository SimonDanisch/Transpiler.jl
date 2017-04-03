module CLTranspiler

using Compat

import ..Transpiler: CIO, symbol_hygiene

using Sugar, OpenCL
using OpenCL: cl
import Sugar: LazyMethod
import Sugar: getsource!, dependencies!, istype, isfunction, getfuncargs, isintrinsic
import Sugar: typename, functionname, show_name, show_type, show_function
import Sugar: supports_overloading, expr_type

const CLMethod = LazyMethod{:CL}

@compat abstract type AbstractCLIO <: CIO end
immutable EmptyCLIO <: AbstractCLIO
end
type CLIO{T <: IO} <: AbstractCLIO
    io::T
    method::CLMethod
end

supports_overloading(io::CLIO) = false



include("intrinsics.jl")
include("printing.jl")
include("rewriting.jl")


immutable CLFunction{Args <: Tuple}
    program::cl.Kernel
    queue::cl.CmdQueue
    method::Nullable{CLMethod}
    source::String
end

_to_cl_types{T}(::Type{T}) = T
_to_cl_types(::Type{Int32}) = Int32
_to_cl_types(::Type{Int64}) = Int32
_to_cl_types(::Type{Float32}) = Float32
_to_cl_types(::Type{Float64}) = Float32
_to_cl_types{T}(arg::T) = _to_cl_types(T)
function _to_cl_types{T <: Union{cl.Buffer, cl.CLArray}}(arg::T)
    return cli.CLArray{eltype(arg), ndims(arg)}
end
function to_cl_types(args::Union{Vector, Tuple})
    map(_to_cl_types, args)
end

immutable EmptyStruct
    # Emtpy structs are not supported in OpenCL, which is why we emit a struct
    # with one floating point field
    x::Float32
    EmptyStruct() = new()
end

function cl_convert{T}(x::T)
    # empty objects are empty and are only usable for dispatch
    isbits(x) && sizeof(x) == 0 && nfields(x) == 0 && return EmptyStruct()
    convert(_to_cl_types(T), x)
end

cl_convert(x::cl.CLArray) = x.buffer # function objects are empty and are only usable for dispatch



const compiled_functions = Dict{Any, CLFunction}()

function empty_compile_cache!()
    empty!(compiled_functions)
    return
end

function CLFunction{T}(f::Function, args::T, queue)
    ctx = cl.context(queue)
    gltypes = to_cl_types(args)
    get!(compiled_functions, (f, gltypes)) do # TODO make this faster
        decl = CLMethod((f, gltypes))
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
        p = cl.build!(
            cl.Program(ctx, source = kernelsource),
            options = "-cl-denorms-are-zero -cl-mad-enable -cl-unsafe-math-optimizations"
        )
        fname = string(functionname(io, decl.signature...))
        k = cl.Kernel(p, fname)
        CLFunction{T}(k, queue, decl, kernelsource)
    end::CLFunction{T}
end

function CLFunction{T}(source_name::Tuple{String, Symbol}, args::T, queue)
    kernelsource, funcname = source_name
    ctx = cl.context(queue)
    p = cl.build!(
        cl.Program(ctx, source = kernelsource),
        options = "-cl-denorms-are-zero -cl-mad-enable -cl-unsafe-math-optimizations"
    )
    k = cl.Kernel(p, string(funcname))
    CLFunction{T}(k, queue, Nullable{CLMethod}(), kernelsource)
end


@generated function (program::CLFunction{T}){T}(
        args::T,
        global_work_size = nothing,
        local_work_size = nothing
    )
    # unrole set_arg! arguments
    unrolled = Expr(:block)
    idx = 0
    args_tuple = Sugar.to_tuple(args)
    for (i, elem) in enumerate(args_tuple)
        if elem <: cl.CLArray
            idx = i
        end
        push!(unrolled.args, :(cl.set_arg!(k, $i, cl_convert(args[$i]))))
    end
    work_dim = if global_work_size <: Void
        if idx != 0
            :(size(args[$idx]))
        else
            error("either supply a global work size, or use a cl.Array to automatically infer global work size")
        end
    else
        :(global_work_size)
    end
    local_size = if local_work_size <: Void
        :(lsize = C_NULL)
    else
        quote
            lsize = Array{Csize_t}(length(gwork))
            for (i, s) in enumerate(local_work_size)
                lsize[i] = s
            end
        end
    end
    quote
        k = program.program
        q = program.queue
        $unrolled
        ret_event = Ref{cl.CL_event}()
        gwork = $work_dim
        gsize = Array{Csize_t}(length(gwork))
        for (i, s) in enumerate(gwork)
            gsize[i] = s
        end
        $local_size
        # TODO support everything from queue() in a performant manner
        cl.@check cl.api.clEnqueueNDRangeKernel(
            q.id, k.id,
            cl.cl_uint(length(gsize)), C_NULL, gsize, lsize,
            cl.cl_uint(0), C_NULL, ret_event
        )
        return cl.Event(ret_event[], retain = false)
    end
end


end
