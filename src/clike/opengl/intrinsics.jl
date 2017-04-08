module GLIntrinsics

using StaticArrays, Sugar

import ..Transpiler: ints, floats, numbers, Numbers, Floats, int, Ints, uchar
import ..Transpiler: fixed_array_length, is_ntuple, is_fixedsize_array, GLMethod
import ..Transpiler: AbstractGLIO, ret, vecs, Vecs, vector_lengths, functions


import Sugar: typename, vecname
using SpecialFunctions: erf

immutable GLArray{T, N} <: AbstractArray{T, N} end
immutable GLTexture{T} <: AbstractArray{T, 1} end

const GLDeviceArray = Union{GLArray, GLTexture}

const Types = Union{vecs..., numbers..., GLArray, GLTexture}
const Functions = Union{map(typeof, functions)...}

glsl_sizeof(T) = sizeof(T) * 8
# for now we disallow Float64 and map it to Float32 -> super hack alert!!!!
glsl_sizeof(::Type{Float64}) = 32
glsl_length{T <: Number}(::Type{T}) = 1
glsl_length(T) = length(T)
prescripts = Dict(
    Float32 => "",
    Float64 => "", # ignore float64 for now
    Int => "i",
    Int32 => "i",
    UInt => "u",
    Bool => "b"
)

function typename{T, N}(io::AbstractGLIO, x::Type{GLArray{T, N}})
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

function vecname{T}(io::AbstractGLIO, t::Type{T})
    N = fixed_array_length(T)
    ET = eltype(T)
    return string(prescripts[ET], "vec", N)
end


imageStore{T}(x::GLArray{T, 1}, i::int, val::NTuple{4, T}) = nothing
imageStore{T}(x::GLArray{T, 2}, i::NTuple{2, int}, val::NTuple{4, T}) = nothing

imageLoad{T}(x::GLArray{T, 1}, i::int) = ret(NTuple{4, T})
imageLoad{T}(x::GLArray{T, 2}, i::NTuple{2, int}) = ret(NTuple{4, T})
imageSize{T, N}(x::GLArray{T, N}) = ret(NTuple{N, int})

const gl_GlobalInvocationID = (0,0,0)

function glintrinsic{F <: Function, T <: Tuple}(f::F, types::Type{T})
    glintrinsic(f, Sugar.to_tuple(types))
end
function glintrinsic{F <: Function}(f::F, types::Tuple)
    # we rewrite Ntuples as glsl arrays, so getindex becomes inbuild
    if f == broadcast
        BF = types[1]
        if BF <: Functions && all(T-> T <: Types, types[2:end])
            return true
        end
    end
    if f == getindex && length(types) == 2 && first(types) <: NTuple && last(types) <: Integer
        return true
    end
    m = methods(f)
    isempty(m) && return false
    sym = first(m).name
    (F <: Functions && all(T-> T <: Types, types)) || (
        # if any intrinsic funtion stub matches
        isdefined(GLIntrinsics, sym) &&
        Base.binding_module(GLIntrinsics, sym) == GLIntrinsics &&
        length(methods(f, types)) == 1
    )
end

end # end GLIntrinsics

using .GLIntrinsics

const gli = GLIntrinsics
import .gli: glintrinsic, GLArray, GLDeviceArray

import Sugar.isintrinsic


function gli.glintrinsic{T}(x::Type{T})
    T <: gli.Types ||
    is_fixedsize_array(T) ||
    T <: Tuple{Numbers} ||
    T <: uchar # uchar in ints makes 0.6 segfault -.-
end
function isintrinsic(x::GLMethod)
    if isfunction(x)
        isintrinsic(Sugar.getfunction(x)) ||
        gli.glintrinsic(x.signature...)
    else
        gli.glintrinsic(x.signature)
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

function glintrinsic{T, I <: gli.Ints}(
        f::typeof(getindex), types::Type{Tuple{T, I}}
    )
    return is_fixedsize_array(T)
end
function glintrinsic{T <: GLDeviceArray, Val, I <: Integer}(
        f::typeof(setindex!), types::Type{Tuple{T, Val, I}}
    )
    return true
end

function glintrinsic(f::typeof(tuple), types::Tuple)
    true
end


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
function Base.getindex{T <: Number}(x::gli.GLArray{T, 2}, idx::NTuple{2, int})
    gli.imageLoad(x, idx)[1]
end
function Base.setindex!{T}(x::gli.GLArray{T, 1}, val::T, i::Integer)
    gli.imageStore(x, i, (val, val, val, val))
end
function Base.setindex!{T}(x::gli.GLArray{T, 2}, val::T, i::Integer, j::Integer)
    setindex!(x, (val, val, val, val), (i, j))
end
function Base.setindex!{T}(x::gli.GLArray{T, 2}, val::T, idx::NTuple{2, int})
    gli.imageStore(x, idx, (val, val, val, val))
end
function Base.setindex!{T}(x::gli.GLArray{T, 2}, val::NTuple{4, T}, idx::NTuple{2, int})
    gli.imageStore(x, idx, val)
end
function Base.setindex!{T}(x::gli.GLArray{T, 1}, val::NTuple{4, T}, i::Integer)
    gli.imageStore(x, i, val)
end
