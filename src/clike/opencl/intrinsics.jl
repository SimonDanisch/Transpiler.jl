module CLIntrinsics

prescripts = Dict(
    Float32 => "",
    Float64 => "", # ignore float64 for now
    Int => "i",
    Int32 => "i",
    UInt => "u",
    Bool => "b"
)
immutable CLArray{T, N} end



function glsl_hygiene(sym)
    # TODO unicode
    # TODO figure out what other things are not allowed
    # TODO startswith gl_, but allow variables that are actually valid inbuilds
    if sym in (:!,)
        return string(sym)
    end
    x = string(sym)
    # this seems pretty hacky! #TODO don't just ignore dots!!!!
    # this is only fine right now, because most opengl intrinsics broadcast anyways
    # but i'm sure this won't hold in general
    x = replace(x, ".", "")
    x = replace(x, "#", "x")
    x = replace(x, "!", "_bang")
    if x == "out"
        x = "_out"
    end
    if x == "in"
        x = "_in"
    end
    x
end



glsl_sizeof(T) = sizeof(T) * 8
# for now we disallow Float64 and map it to Float32 -> super hack alert!!!!
glsl_sizeof(::Type{Float64}) = 32
glsl_length{T <: Number}(::Type{T}) = 1
glsl_length(T) = length(T)


# Number types
# Abstract types
# for now we use Int, more accurate would be Int32. But to make things simpler
# we rewrite Int to Int32 implicitely like this!
const int = Int
# same goes for float
const float = Float64
const uint = UInt

const ints = (int, Int32, uint)
const floats = (Float32, float)
const numbers = (ints..., floats..., Bool)

const Ints = Union{ints...}
const Floats = Union{floats...}
const Numbers = Union{numbers...}



using StaticArrays
_vecs = []
for i = 2:4, T in numbers
    push!(_vecs, NTuple{i, T})
    push!(_vecs, SVector{i, T})
end

const vecs = (_vecs...)
const Vecs = Union{vecs...}
const Types = Union{vecs..., numbers..., CLArray}

@noinline function ret{T}(::Type{T})::T
    unsafe_load(Ptr{T}(C_NULL))
end

# intrinsics not defined in Base need a function stub:
for i = 2:4
    @eval begin
        function (::Type{NTuple{$i, T}}){T <: Numbers, N, T2 <: Numbers}(x::NTuple{N, T2})
            ntuple(i-> T(x[i]), Val{$i})
        end
    end
end






glsl_name(x) = Symbol(glsl_hygiene(_glsl_name(x)))
_glsl_name(T::QuoteNode) = _glsl_name(T.value)

function _glsl_name(T)
    str = if isa(T, Expr) && T.head == :curly
        string(T, "_", join(T.args, "_"))
    elseif isa(T, Symbol)
        string(T)
    elseif isa(T, Tuple)
        str = "Tuple_"
        if !isempty(t.parameters)
            tstr = map(typename, t.parameters)
            str *= join(tstr, "_")
        end
        str
    elseif isa(T, Type)
        str = string(T.name.name)
        if !isempty(T.parameters)
            tstr = map(T.parameters) do t
                if isa(t, DataType)
                    typename(t)
                else
                    string(t)
                end
            end
            str *= string("_", join(tstr, "_"))
        end
        str
    else
        error("Not transpilable: $T")
    end
    return str
end

function _glsl_name{T, N}(x::Type{CLArray{T, N}})
    if !(N in (1, 2, 3))
        # TODO, fake ND arrays with 1D array
        error("GPUArray can't have more than 3 dimensions for now")
    end
    tname = typename(T)
    "__global $tname *"
end


function _glsl_name{T <: Vecs}(t::Type{T})
    N = if T <: Tuple
        length(T.parameters)
    else
        length(T)
    end
    return string('(', typename(eltype(T)), N, ')')
end
function _glsl_name{N, T}(t::Type{NTuple{N, T}})
    println(t)
    # Rewrite remaining rewrite ntuples as glsl arrays
    # e.g. float[3]
    string(typename(T), '[', N, ']')
end
function _glsl_name{N, T}(t::Type{SVector{N, T}})
    println(t)
    # Rewrite remaining rewrite ntuples as glsl arrays
    # e.g. float[3]
    string(typename(T), '[', N, ']')
end
_glsl_name(x::Union{AbstractString, Symbol}) = x



_glsl_name(::typeof(^)) = "pow"
if VERSION < v"0.6"
    _glsl_name(::typeof(.+)) = "+"
    _glsl_name(::typeof(.-)) = "-"
    _glsl_name(::typeof(.*)) = "*"
    _glsl_name(::typeof(./)) = "/"
end

_glsl_name(x::Type{Void}) = "void"
_glsl_name(x::Type{Float64}) = "float"
_glsl_name(x::Type{Float32}) = "float"
_glsl_name(x::Type{Int}) = "int"
_glsl_name(x::Type{Int32}) = "int"
_glsl_name(x::Type{UInt}) = "uint"
_glsl_name(x::Type{Bool}) = "bool"
_glsl_name{T}(x::Type{Ptr{T}}) = "$(typename(T)) *"

# TODO this will be annoying on 0.6
# _glsl_name(x::typeof(cli.:(*))) = "*"
# _glsl_name(x::typeof(cli.:(<=))) = "lessThanEqual"
# _glsl_name(x::typeof(cli.:(+))) = "+"

function _glsl_name{F <: Function}(f::Union{F, Type{F}})
    # Taken from base... #TODO make this more stable
    _glsl_name(F.name.mt.name)
end

typename{T}(::Type{T}) = glsl_name(T)
global operator_replacement
let _operator_id = 0
    const operator_replace_dict = Dict{Char, String}()
    function operator_replacement(char)
        get!(operator_replace_dict, char) do
            _operator_id += 1
            string("op", _operator_id)
        end
    end
end

function typename{T <: Function}(::Type{T})
    x = string(T)
    x = replace(x, ".", "_")
    x = sprint() do io
        for char in x
            if Base.isoperator(Symbol(char))
                print(io, operator_replacement(char))
            else
                print(io, char)
            end
        end
    end
    glsl_name(x)
end

#typealias for inbuilds
for i = 2:4, T in numbers
    nvec = NTuple{i, T}
    name = glsl_name(nvec)
    if !isdefined(name)
        @eval const $name = $nvec
    end
end

get_global_id(dim::int) = ret(int)


#######################################
# globals
const functions = (
    +, -, *, /, ^, <=, .<=,
    sin, tan, sqrt, cos, mod, floor
)

const Functions = Union{map(typeof, functions)...}


function glintrinsic{F <: Function, T <: Tuple}(f::F, types::Type{T})
    glintrinsic(f, (T.parameters...))
end
function glintrinsic{F <: Function}(f::F, types::Tuple)
    # we rewrite Ntuples as glsl arrays, so getindex becomes inbuild
    if f == getindex && length(types) == 2 && first(types) <: NTuple && last(types) <: Integer
        return true
    end
    if f == getindex && length(types) == 2 && first(types) <: CLArray && last(types) <: Integer
        return true
    end
    m = methods(f)
    isempty(m) && return false
    sym = first(m).name
    (F <: Functions && all(T-> T <: Types, types)) || (
        # if any intrinsic funtion stub matches
        isdefined(CLIntrinsics, sym) &&
        Base.binding_module(CLIntrinsics, sym) == CLIntrinsics &&
        length(methods(f, types)) == 1
    )
end

end # end CLIntrinsics


using .CLIntrinsics

const cli = CLIntrinsics
import .cli: glsl_name, typename, image_format, glintrinsic, CLArray


####################################
# Be a type pirate on 0.5!
# We shall turn this package into 0.6 only, but 0.6 is broken right now
# so that's why we need pirating!
if VERSION < v"0.6"
    Base.broadcast{N}(f, a::NTuple{N, Any}, b::NTuple{N, Any}) = map(f, a, b)
    Base.broadcast{N}(f, a::NTuple{N, Any}) = map(f, a)
    Base.:(.<=){N}(a::NTuple{N, Any}, b::NTuple{N, Any}) = map(<=, a, b)
    Base.:(.*){N}(a::NTuple{N, Any}, b::NTuple{N, Any}) = map(*, a, b)
    Base.:(.+){N}(a::NTuple{N, Any}, b::NTuple{N, Any}) = map(+, a, b)
end

import Sugar.isintrinsic

function isintrinsic(x::GLMethod)
    if isfunction(x)
        isintrinsic(Sugar.getfunction(x)) ||
        cli.glintrinsic(x.signature...)
    else
        x.signature <: cli.Types
    end
end

# copied from rewriting. TODO share implementation!

# Make constructors inbuild for now. TODO, only make default constructors inbuild
function glintrinsic{T}(f::Type{T}, types::ANY)
    return true
end
# homogenous tuples, translated to glsl array
function glintrinsic{N, T, I <: Integer}(
        f::typeof(getindex), types::Type{Tuple{NTuple{N, T}, I}}
    )
    return true
end

function glintrinsic{T <: cli.Vecs, I <: cli.int}(
        f::typeof(getindex), types::Type{Tuple{T, I}}
    )
    return true
end
function glintrinsic{T <: CLArray, Val, I <: Integer}(
        f::typeof(setindex!), types::Type{Tuple{T, Val, I}}
    )
    return true
end


function glintrinsic{V1 <: cli.Vecs, V2 <: cli.Vecs}(
        f::Type{V1}, types::Type{Tuple{V2}}
    )
    return true
end
function glintrinsic(f::typeof(tuple), types::ANY)
    true
end


function glintrinsic(f::typeof(broadcast), types::ANY)
    tuptypes = (types.parameters...)
    F = tuptypes[1]
    if F <: cli.Functions && all(T-> T <: cli.Types, tuptypes[2:end])
        return true
    end
    false
end

function Base.getindex{T, N}(a::CLArray{T, N}, id::Integer)
    cli.ret(T)
end
function Base.setindex!(a::CLArray, id::Integer)
    nothing
end
