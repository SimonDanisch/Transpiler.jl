
import Sugar: LazyMethod, supports_overloading, expr_type
import Sugar: getsource!, dependencies!, istype, isfunction, getfuncargs, isintrinsic
import Sugar: typename, functionname, show_name, show_type, show_function

const GLMethod = LazyMethod{:GL}

@compat abstract type AbstractGLIO <: CIO end
immutable EmptyGLIO <: AbstractGLIO
end
immutable GLFunction{Args <: Tuple}
    program::GLProgram
    local_size::NTuple{3, Int}
end
type GLIO{T <: IO} <: AbstractGLIO
    io::T
    method::GLMethod
end

supports_overloading(io::GLIO) = true

include("intrinsics.jl")
include("printing.jl")
include("rewriting.jl")



#_to_gl_types{T}(::Type{T}) = T
_to_gl_types{T}(::T) = T
_to_gl_types{T}(::Type{T}) = Type{T}
_to_gl_types(::Int64) = Int32
_to_gl_types(::Float64) = Float32

function _to_gl_types{T <: Union{GLBuffer, Texture}}(arg::T)
    return gli.CLArray{eltype(arg), ndims(arg)}
end
function to_gl_types(args::Union{Vector, Tuple})
    map(_to_gl_types, args)
end



function gl_convert{T}(x::T)
    # empty objects are empty and are only usable for dispatch
    isbits(x) && sizeof(x) == 0 && nfields(x) == 0 && return EmptyStruct()
    # same applies for types
    isa(x, Type) && return EmptyStruct()
    convert(_to_gl_types(x), x)
end

const gl_compiled_functions = Dict{Any, GLFunction}()

function gl_empty_compile_cache!()
    empty!(gl_compiled_functions)
    return
end

function GLFunction{T}(f::Function, args::T, window)
    gltypes = to_gl_types(args)
    get!(gl_compiled_functions, (f, gltypes)) do # TODO make this faster
        decl = GLFunction((f, gltypes))
        funcsource = getsource!(decl)
        # add compute program dependant infos
        io = GLIO(IOBuffer(), decl)
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
        println(io, funcsource)
        funcargs = getfuncargs(decl)
        declare_global(io, getfuncargs(decl))
        varnames = map(x-> string(global_identifier, x.args[1]), funcargs)
        print(io, "void main(){\n    ")
        show_name(io, f)
        print(io, "(", join(varnames, ", "), ");\n}")
        shader = GLAbstraction.compile_shader(take!(io.io), GL_COMPUTE_SHADER, Symbol(f))
        program = GLAbstraction.compile_program([shader], [])
        GLFunction{T}(program, local_size)
    end::GLFunction{T}
end

# function GLFunction{T}(source_name::Tuple{String, Symbol}, args::T, queue)
#     kernelsource, funcname = source_name
#     ctx = cl.context(queue)
#     p = cl.build!(
#         cl.Program(ctx, source = kernelsource),
#         options = "-cl-denorms-are-zero -cl-mad-enable -cl-unsafe-math-optimizations"
#     )
#     k = cl.Kernel(p, string(funcname))
#     GLFunction{T}(k, queue, Nullable{GLFunction}(), kernelsource)
# end



useprogram(p::GLFunction) = glUseProgram(p.program.id)


function (p::GLFunction{Args}){Args}(args::Args, size::NTuple{3})
    useprogram(p)
    for i = 1:length(args)
        bindlocation(args[i], i-1)
    end
    size = ntuple(Val{3}) do i
        ceil(Int, size[i] / p.local_size[i])
    end
    glDispatchCompute(size[1], size[2], size[3])
    return
end
