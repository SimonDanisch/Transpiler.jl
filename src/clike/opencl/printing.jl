import Base: show

using Sugar: LazyMethod, typename, functionname, print_dependencies, getsource!
import Sugar: supports_overloading

const CLMethod = LazyMethod{:CL}

@compat abstract type AbstractCLIO <: CIO end

immutable EmptyCLIO <: AbstractCLIO end

type CLIO{T <: IO} <: AbstractCLIO
    io::T
    method::CLMethod
end

(::Type{CIO})(io, x::CLMethod) = CLIO(io, x)

supports_overloading(io::CLIO) = false

include("intrinsics.jl")
include("rewriting.jl")

Sugar.is_tracked_type(m, ::Type{<: cli.CLArray}) = true

"""
Transpiles the source of a given function `f` for it's argument types `args` (Tuple{<})
"""
function kernel_source(f::Function, args::NTuple{N, <: DataType}) where N
    method = CLMethod((f, args))
    Sugar.track_types!(method)
    funcsource = getsource!(method)
    # add compute program dependant infos
    io = CLIO(IOBuffer(), method)
    println(io, "// dependencies")
    visited = Set()
    for dep in dependencies!(method)
        print_dependencies(io, dep, visited) # this is recursive but would print method itself
    end
    println(io, "// ########################")
    println(io, "// Main inner function")
    println(io, "// ", method.signature)
    print(io, "__kernel ") # mark as kernel function
    println(io, funcsource)
    fname = string(functionname(io, method))
    String(take!(io.io)), method, fname
end


function show(io::CLIO, x::Union{Float32, Float64})
    if isinf(x)
        print(io, "INFINITY")
    elseif isnan(x)
        print(io, "NAN")
    elseif isa(x, Float32)
        print(io, Float64(x), 'f')
    else
        print(io, Float64(x))
    end
end


function Sugar.gettypesource(x::CLMethod)
    T = x.signature
    # TODO, do something about void! It's the same as the intrinsic void for return types
    # but gets used completely different in Julia -E.g. in OpenCL it's not possible to have an instance of type void
    tname = typename(EmptyCLIO(), T)

    str = if (!isleaftype(T) || T <: Type{X} where X) # emit type instances as singletons
        """typedef int $tname; // placeholder type instance
        __constant $tname TYP_INST_$tname = 0;
        """
    else
        if T <: Symbol # ignore for now?
            return ""
        elseif isleaftype(T) && sizeof(T) == 0 && nfields(T) == 0
            # emit empty types as Int32, since struct can't be empty
            str = "typedef int $tname; // empty type emitted as an int"
            if T <: Sugar.AllFuncs
                str = "__constant int FUNC_INST_$tname = 0;\n" * str
            end
            return str
        else
            sprint() do io
                print(io, "struct  __attribute__ ((packed)) TYP$tname{\n")
                nf = nfields(T)
                fields = []
                if nf == 0 # structs can't be empty
                    # we use float as a short placeholder type.
                    # TODO, are there cases where float is no good?
                    println(io, "float empty; // structs can't be empty")
                else
                    for i in 1:nf
                        FT = fieldtype(T, i)
                        print(io, "    ", typename(EmptyCLIO(), FT))
                        print(io, ' ')
                        print(io, c_fieldname(x, T, i))
                        println(io, ';')
                    end
                end
                println(io, "};")
                println(io, "typedef struct TYP$tname $tname;")
            end
        end
    end
    return str
end


fixed_size_array_fieldname(::CLMethod, T, i) = "s$(i-1)"
