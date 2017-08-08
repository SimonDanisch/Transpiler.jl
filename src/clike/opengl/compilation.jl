import Sugar: LazyMethod, supports_overloading, expr_type
import Sugar: getsource!, dependencies!, istype, isfunction, getfuncargs, isintrinsic
import Sugar: typename, functionname, show_name, show_type, show_function

const GLMethod = LazyMethod{:GL}
const GEOMMethod = LazyMethod{:GEOM}
const GLMethods = Union{GLMethod, GEOMMethod}

@compat abstract type AbstractGLIO <: CIO end

immutable GLFunction{Args <: Tuple}
    program::GLProgram
    local_size::NTuple{3, Int}
end
type GLIO{T <: IO} <: AbstractGLIO
    io::T
    method::GLMethods
end


function (::Type{CIO})(io, m::GLMethod)
    GLIO(io, m)
end

supports_overloading(io::GLIO) = true

include("intrinsics.jl")
include("printing.jl")
include("rewriting.jl")



#to_gl_types{T}(::Type{T}) = T
to_gl_types{T}(::T) = to_gl_types(T)
to_gl_types{T}(::Type{T}) = T
to_gl_types{T}(::Type{Type{T}}) = Type{T}
to_gl_types(::Type{Int64}) = Int32
to_gl_types(::Type{Float64}) = Float32

function to_gl_types{T <: Texture}(arg::Type{T})
    return gli.GLTexture{eltype(arg), ndims(arg)}
end
function to_gl_types{T <: GLBuffer}(arg::Type{T})
    return gli.GLArray{eltype(arg), ndims(arg)}
end

function gl_convert{T}(x::T)
    # empty objects are empty and are only usable for dispatch
    isbits(x) && sizeof(x) == 0 && nfields(x) == 0 && return EmptyStruct()
    # same applies for types
    isa(x, Type) && return EmptyStruct()
    convert(to_gl_types(x), x)
end

const gl_compiled_functions = Dict{Any, GLFunction}()

function gl_empty_compile_cache!()
    empty!(gl_compiled_functions)
    return
end

function GLFunction{T}(f::Function, args::T, window)
    gltypes = to_gl_types.(args)
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
