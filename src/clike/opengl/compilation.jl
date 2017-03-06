module GLSLTranspiler

using Sugar, GeometryTypes, FileIO
using GLWindow, GLAbstraction, ModernGL, Reactive, GLFW, GeometryTypes
import Sugar: ssavalue_name, ASTIO, get_slottypename, get_type, LazyMethod
import Sugar: getsource!, dependencies!, istype, isfunction, getfuncargs, isintrinsic

const GLMethod = LazyMethod{:GL}

include("intrinsics.jl")
include("printing.jl")
include("rewriting.jl")

immutable ComputeProgram{Args <: Tuple}
    program::GLProgram
    local_size::NTuple{3, Int}
end

_to_glsl_types(::Type{Int32}) = Int32
_to_glsl_types(::Type{Int64}) = Int32
_to_glsl_types(::Type{Float32}) = Float32
_to_glsl_types(::Type{Float64}) = Float32
_to_glsl_types{T}(arg::T) = _to_glsl_types(T)
_to_glsl_types{T}(::Type{T}) = T

function _to_glsl_types{T <: Texture}(arg::T)
    return gli.GLArray{eltype(arg), ndims(arg)}
end
function to_glsl_types(args::Union{Vector, Tuple})
    map(_to_glsl_types, args)
end

const compiled_functions = Dict{Any, ComputeProgram}()

function ComputeProgram{T}(f::Function, args::T; local_size = (16, 16, 1))
    gltypes = to_glsl_types(args)
    get!(compiled_functions, (f, gltypes)) do # TODO make this faster
        decl = GLMethod((f, gltypes))
        funcsource = getsource!(decl)
        # add compute program dependant infos
        io = GLSLIO(IOBuffer(), decl)
        print(io,
            "#version 430\n", # hardcode version for now #TODO don't hardcode :P
            "layout (local_size_x = 16, local_size_y = 16) in;\n", # same here
        )

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
        ComputeProgram{T}(program, local_size)
    end::ComputeProgram{T}
end

useprogram(p::ComputeProgram) = glUseProgram(p.program.id)


function (p::ComputeProgram{Args}){Args}(args::Args, size::NTuple{3})
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


end
