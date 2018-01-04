module GLIntrinsics

using StaticArrays, Sugar, GeometryTypes, MacroTools, GLAbstraction

import ..Transpiler: ints, floats, numbers, Numbers, Floats, int, Ints, uchar
import ..Transpiler: fixed_array_length, is_ntuple, is_fixedsize_array, GLMethod
import ..Transpiler: AbstractGLIO, ret, vecs, Vecs, vector_lengths, functions

import Sugar: typename, vecname
using SpecialFunctions: erf

immutable GLArray{T, N} <: AbstractArray{T, N} end
immutable GLTexture{T, N} <: AbstractArray{T, N} end

const GLDeviceArray = Union{GLArray, GLTexture}
const Types = Union{vecs..., numbers..., GLArray, GLTexture}
const Functions = Union{map(typeof, functions)...}


const intrinsic_signatures = Dict{Function, Any}()

macro gl_intrinsic(expr)
    matched = @capture(
        expr,
        (func_(args__) where {T__} = body_) | (func_(args__) = body_)
    )
    @assert matched "internal error: intrinsic wasn't matched: $expr"
    ret_expr = Expr(:block)

    # it's possible to define methods in base as intrinsic.
    # if they're not in base, we need to define a function stub
    if !isdefined(Base, func)
        push!(ret_expr.args, esc(expr))
    end
    types = map(args) do arg
        @assert isa(arg, Expr) && arg.head == :(::) "wrong type declaration"
        arg.args[end]
    end
    tuple_typ = :(Tuple{$(types...)})
    if T != nothing
        tuple_typ = :($tuple_typ where {$(T...)})
    end
    push!(ret_expr.args, :(intrinsic_signatures[$func] = $tuple_typ))
    ret_expr
end


@gl_intrinsic imageStore(x::GLArray{T, 1}, i::int, val::NTuple{4, T}) where T = nothing
@gl_intrinsic imageStore(x::GLArray{T, 2}, i::NTuple{2, int}, val::NTuple{4, T}) where T = nothing

@gl_intrinsic imageLoad(x::GLArray{T, 1}, i::int) where T = ret(NTuple{4, T})
@gl_intrinsic imageLoad(x::GLArray{T, 2}, i::NTuple{2, int}) where T = ret(NTuple{4, T})

@gl_intrinsic texelFetch(x::GLTexture{T, 1}, i::int, lod::int) where T = ret(NTuple{4, T})
@gl_intrinsic texelFetch(x::GLTexture{T, 2}, i::NTuple{2, int}, lod::int) where T = ret(NTuple{4, T})

@gl_intrinsic texture(x::GLTexture{T, 1}, i::Float32) where T = ret(NTuple{4, T})
@gl_intrinsic texture(x::GLTexture{T, 2}, i::Vec2f0) where T = ret(NTuple{4, T})
@gl_intrinsic texture(x::GLTexture{T, 3}, i::Vec3f0) where T = ret(NTuple{4, T})

@gl_intrinsic imageSize(x::GLArray{T, N}) where {T, N} = ret(NTuple{N, int})
@gl_intrinsic textureSize(::GLTexture{T, N}) where {T, N} = ret(NTuple{N, int})


#=
Gradient in x direction
This is sadly a bit hard to implement for a pure CPU versions, since it's pretty much backed into the GPU hardware.
How it seems to work is, that it takes the values from neighboring registers, which work in parallel on the pixels
of the triangle, so they actually do hold the neighboring values needed to calculate the gradient.
=#
@gl_intrinsic dFdx(value::T) where T = T(0.001) # just default to a small gradient if it's called on the CPU
@gl_intrinsic dFdy(value::T) where T = T(0.001) # just default to a small gradient if it's called on the CPU


@gl_intrinsic EmitVertex() = nothing
@gl_intrinsic EndPrimitive() = nothing

const gl_GlobalInvocationID = Vec3f0(0, 0, 0)
const gl_FragCoord = Vec4f0(0, 0, 0, 0)

end # end GLIntrinsics

using .GLIntrinsics
using GeometryTypes
const gli = GLIntrinsics
import .gli: GLArray, GLDeviceArray, GLTexture

glsl_sizeof(::T) where T = sizeof(T) * 8
# for now we disallow Float64 and map it to Float32 -> super hack alert!!!!
glsl_sizeof(::Type{Float64}) = 32
glsl_length(::Type{T}) where T <: Number = 1
glsl_length(T) = length(T)

prescripts = Dict(
    Float32 => "",
    Float64 => "", # ignore float64 for now
    Int => "i",
    Int32 => "i",
    UInt => "u",
    Bool => "b"
)

function typename{T, N}(io::AbstractGLIO, x::Type{GLTexture{T, N}})
    string(prescripts[eltype(T)], "sampler", N, "D")
end
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
function typename{T <: SMatrix}(io::AbstractGLIO, ::Type{T})
    M, N = size(T)
    string(prescripts[eltype(T)], "mat", M == N ? M : string(M, "x", N))
end

function Sugar.vecname{T}(io::AbstractGLIO, t::Type{T})
    N = fixed_array_length(T)
    ET = eltype(T)
    return string(prescripts[ET], "vec", N)
end



import Sugar.isintrinsic

function is_native_type(m::GLMethod, T)
    T <: gli.Types || is_fixedsize_array(m, T) || T <: Tuple{T} where T <: cli.Numbers
end
function backend_intrinsic(m::GLMethod, func::ANY, sig_tuple::ANY)
    haskey(gli.intrinsic_signatures, func) || return false
    sig = gli.intrinsic_signatures[func]
    sig_tuple <: sig
end

import Base: getindex, setindex!, size

GlobalInvocationID() = gli.gl_GlobalInvocationID
FragCoord() = gli.gl_FragCoord

size{T, N}(x::gli.GLArray{T, N}) = gli.imageSize(x)
size{T, N}(x::gli.GLTexture{T, N}) = gli.textureSize(x)

getindex{T}(x::gli.GLTexture{T, 1}, i::Integer) = gli.texelFetch(x, i, 0)
getindex{T}(x::gli.GLTexture{T, 1}, i::Integer, j::Integer) = gli.texelFetch(x, (i, j), 0)

getindex{T}(x::gli.GLTexture{T, 1}, i::AbstractFloat) = gli.texture(x, i)
getindex{T}(x::gli.GLTexture{T, 2}, i::AbstractFloat, j::AbstractFloat) = gli.texture(x, Vec2f0(i, j))
getindex{T}(x::gli.GLTexture{T, 2}, idx::Vec2f0) = gli.texture(x, idx)
getindex{T}(x::gli.GLTexture{T, 3}, idx::Vec3f0) = gli.texture(x, idx)

getindex{T}(x::gli.GLArray{T, 1}, i::Integer) = gli.imageLoad(x, i)
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

function gl_erf{T <: AbstractFloat}(x::T)
    # constants
    a1 =  T(0.254829592)
    a2 = T(-0.284496736)
    a3 =  T(1.421413741)
    a4 = T(-1.453152027)
    a5 =  T(1.061405429)
    p  =  T(0.3275911)

    # Save the sign of x
    sign = 1
    if (x < 0)
        sign = -1
    end
    xabs = abs(x)
    # A&S formula 7.1.26
    t = T(1.0) / (T(1.0) + p * xabs)
    y = T(1.0) - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t*exp(-xabs*xabs)

    return sign * y
end
gl_erfc{T <: AbstractFloat}(x::T) = T(1.0) - gl_erf(x)

# FMA is only supported in opengl >= 4.0 . To keep it simple, we just use this fallback for now
# until we propagate opengl versions and emit code accordingly
gl_fma{T <: AbstractFloat}(a::T, b::T, c::T) = a * b + c
