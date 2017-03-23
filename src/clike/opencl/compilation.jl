module CLTranspiler

import ..Transpiler: CIO, symbol_hygiene

using Sugar, OpenCL
using OpenCL: cl
import Sugar: ssavalue_name, ASTIO, get_slottypename, get_type, LazyMethod
import Sugar: getsource!, dependencies!, istype, isfunction, getfuncargs, isintrinsic
import Sugar: isintrinsic, typename, functionname, show_name, show_type, show_function
import Sugar: supports_overloading, show_function, expr_type

const CLMethod = LazyMethod{:CL}

abstract AbstractCLIO <: CIO
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

immutable ComputeProgram{Args <: Tuple}
    program::cl.Kernel
    queue::cl.CmdQueue
    method::CLMethod
    source::String
end

_to_cl_types(::Type{Int32}) = Int32
_to_cl_types(::Type{Int64}) = Int32
_to_cl_types(::Type{Float32}) = Float32
_to_cl_types(::Type{Float64}) = Float32
_to_cl_types{T}(arg::T) = _to_cl_types(T)
_to_cl_types{T}(::Type{T}) = T

function _to_cl_types{T <: cl.Buffer}(arg::T)
    return cli.CLArray{eltype(arg), ndims(arg)}
end
function to_cl_types(args::Union{Vector, Tuple})
    map(_to_cl_types, args)
end

const compiled_functions = Dict{Any, ComputeProgram}()

function ComputeProgram{T}(f::Function, args::T, queue)
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
        p = cl.build!(cl.Program(ctx, source = kernelsource), raise = false)
        success = true
        for (dev, status) in cl.info(p, :build_status)
            if status == cl.CL_BUILD_ERROR
                println("Couldn't compile: ")
                println(kernelsource)
                error(cl.info(p, :build_log)[dev])
            end
        end
        fname = string(functionname(io, decl.signature...))
        k = cl.Kernel(p, fname)
        ComputeProgram{T}(k, queue, decl, kernelsource)
    end::ComputeProgram{T}
end

cl_convert(x) = x
cl_convert(x::Function) = 0f0 # function objects are empty and are only usable for dispatch
cl_convert(x::cl.CLArray) = x.buffer # function objects are empty and are only usable for dispatch

function (program::ComputeProgram{T}){T}(args::T;
        global_work_size = nothing,
        local_work_size = nothing
    )
    if global_work_size == nothing
        for elem in args # search of a opencl buffer
            if isa(elem, cl.CLArray)
                global_work_size = size(elem)
                break
            end
        end
    end
    args_conv = map(cl_convert, args)
    program.queue(
        program.program,
        global_work_size, local_work_size,
        args_conv...
    )
end


end
