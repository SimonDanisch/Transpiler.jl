module GLSLIntrinsics

prescripts = Dict(
    Float32 => "",
    Float64 => "", # ignore float64 for now
    Int => "i",
    Int32 => "i",
    UInt => "u",
    Bool => "b"
)
immutable GLArray{T, N} end

function image_format{T, N}(x::Type{GLArray{T, N}})
    # TODO add other fromats
    error("Element type $T not implemented yet")
end
function image_format{N}(x::Type{GLArray{Float32, N}})
    # TODO add other fromats
    "r32f"
end



function glsl_hygiene(sym)
    # TODO unicode
    # TODO figure out what other things are not allowed
    # TODO startswith gl_, but allow variables that are actually valid inbuilds
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
const Types = Union{vecs..., numbers..., GLArray}

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

function _glsl_name{T, N}(x::Type{GLArray{T, N}})
    if !(N in (1, 2, 3))
        # TODO, fake ND arrays with 1D array
        error("GPUArray can't have more than 3 dimensions for now")
    end
    sz = glsl_sizeof(T)
    len = glsl_length(T)
    if true
        qualifiers = [image_format(x)]
        string("layout (", join(qualifiers, ", "), ") image$(N)D")
    else
        "image$(N)D$(len)x$(sz)_bindless"
    end
end


function _glsl_name{T <: Vecs}(t::Type{T})
    N = if T <: Tuple
        length(T.parameters)
    else
        length(T)
    end
    return string(prescripts[eltype(T)], "vec", N)
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

# TODO this will be annoying on 0.6
# _glsl_name(x::typeof(gli.:(*))) = "*"
# _glsl_name(x::typeof(gli.:(<=))) = "lessThanEqual"
# _glsl_name(x::typeof(gli.:(+))) = "+"

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

imageStore{T}(x::GLArray{T, 1}, i::int, val::NTuple{4, T}) = nothing
imageStore{T}(x::GLArray{T, 2}, i::ivec2, val::NTuple{4, T}) = nothing

imageLoad{T}(x::GLArray{T, 1}, i::int) = ret(NTuple{4, T})
imageLoad{T}(x::GLArray{T, 2}, i::ivec2) = ret(NTuple{4, T})
imageSize{T, N}(x::GLArray{T, N}) = ret(NTuple{N, int})


#######################################
# globals
const gl_GlobalInvocationID = uvec3((0,0,0))
const functions = (
    +, -, *, /, ^, <=, .<=,
    sin, tan, sqrt, cos, mod, floor, isinf, isnan,
    imageLoad, imageSize, imageStore
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
    m = methods(f)
    isempty(m) && return false
    sym = first(m).name
    (F <: Functions && all(T-> T <: Types, types)) || (
        # if any intrinsic funtion stub matches
        isdefined(GLSLIntrinsics, sym) &&
        Base.binding_module(GLSLIntrinsics, sym) == GLSLIntrinsics &&
        length(methods(f, types)) == 1
    )
end

end # end GLSLIntrinsics


using .GLSLIntrinsics

const gli = GLSLIntrinsics
import .gli: glsl_name, typename, image_format, glintrinsic

function GlobalInvocationID()
    gli.gl_GlobalInvocationID
end

function Base.size{T, N}(x::gli.GLArray{T, N})
    gli.imageSize(x)
end
function Base.getindex{T}(x::gli.GLArray{T, 1}, i::Integer)
    gli.imageLoad(x, i)
end
function Base.getindex{T}(x::gli.GLArray{T, 2}, i::Integer, j::Integer)
    getindex(x, (i, j))
end
function Base.getindex{T <: Number}(x::gli.GLArray{T, 2}, idx::gli.ivec2)
    gli.imageLoad(x, idx)[1]
end
function Base.setindex!{T}(x::gli.GLArray{T, 1}, val::T, i::Integer)
    gli.imageStore(x, i, (val, val, val, val))
end
function Base.setindex!{T}(x::gli.GLArray{T, 2}, val::T, i::Integer, j::Integer)
    setindex!(x, (val, val, val, val), (i, j))
end
function Base.setindex!{T}(x::gli.GLArray{T, 2}, val::T, idx::gli.ivec2)
    gli.imageStore(x, idx, (val, val, val, val))
end
function Base.setindex!{T}(x::gli.GLArray{T, 2}, val::NTuple{4, T}, idx::gli.ivec2)
    gli.imageStore(x, idx, val)
end
function Base.setindex!{T}(x::gli.GLArray{T, 1}, val::NTuple{4, T}, i::Integer)
    gli.imageStore(x, i, val)
end

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
        gli.glintrinsic(x.signature...)
    else
        x.signature <: gli.Types
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

function glintrinsic{T <: gli.Vecs, I <: gli.int}(
        f::typeof(getindex), types::Type{Tuple{T, I}}
    )
    return true
end

function glintrinsic{V1 <: gli.Vecs, V2 <: gli.Vecs}(
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
    if F <: gli.Functions && all(T-> T <: gli.Types, tuptypes[2:end])
        return true
    end
    false
end
