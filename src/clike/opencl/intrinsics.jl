module CLIntrinsics
import ..CLTranspiler: AbstractCLIO, EmptyCLIO
import Sugar: typename, vecname

immutable CLArray{T, N} <: AbstractArray{T, N} end


# Number types
# Abstract types
# for now we use Int, more accurate would be Int32. But to make things simpler
# we rewrite Int to Int32 implicitely like this!
const int = Int
# same goes for float
const float = Float64
const uint = UInt

const ints = (int, Int32, uint, Int64)
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


function typename{T, N}(io::AbstractCLIO, x::Type{CLArray{T, N}})
    if !(N in (1, 2, 3))
        # TODO, fake ND arrays with 1D array
        error("GPUArray can't have more than 3 dimensions for now")
    end
    tname = typename(io, T)
    "__global $tname *"
end

function vecname{T <: Vecs}(io::AbstractCLIO, t::Type{T})
    N = if T <: Tuple
        length(T.parameters)
    else
        length(T)
    end
    return string(typename(io, eltype(T)), N)
end

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

#typealias for inbuilds
for i = 2:4, T in numbers
    nvec = NTuple{i, T}
    name = Symbol(vecname(EmptyCLIO(), nvec))
    if !isdefined(name)
        @eval const $name = $nvec
    end
end

get_global_id(dim::int) = ret(int)

pow{T <: Numbers}(a::T, b::T) = ret(T)
#######################################
# globals
const functions = (
    +, -, *, /, ^, <=, .<=, !, <, >, ==, !=, |, &,
    sin, tan, sqrt, cos, mod, floor, log, atan2, max, min,
    abs, pow
)

const Functions = Union{map(typeof, functions)...}

function clintrinsic{F <: Function, T <: Tuple}(f::F, types::Type{T})
    clintrinsic(f, (T.parameters...))
end
function clintrinsic{F <: Function}(f::F, types::Tuple)
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
import .cli: clintrinsic, CLArray


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

is_fixedsize_array(x) = false
is_fixedsize_array{N, T}(::Type{NTuple{N, T}}) = isleaftype(T)
function cli.clintrinsic{T}(x::Type{T})
    T <: cli.Types ||
    is_fixedsize_array(T)
end
function isintrinsic(x::CLMethod)
    if isfunction(x)
        isintrinsic(Sugar.getfunction(x)) ||
        cli.clintrinsic(x.signature...)
    else
        cli.clintrinsic(x.signature)
    end
end

# copied from rewriting. TODO share implementation!

# Make constructors inbuild for now. TODO, only make default constructors inbuild
function clintrinsic{T}(f::Type{T}, types::ANY)
    return true
end
# homogenous tuples, translated to glsl array
function clintrinsic{N, T, I <: Integer}(
        f::typeof(getindex), types::Type{Tuple{NTuple{N, T}, I}}
    )
    return true
end

function clintrinsic{T <: cli.Vecs, I <: cli.int}(
        f::typeof(getindex), types::Type{Tuple{T, I}}
    )
    return true
end
function clintrinsic{T <: CLArray, Val, I <: Integer}(
        f::typeof(setindex!), types::Type{Tuple{T, Val, I}}
    )
    return true
end


function clintrinsic{V1 <: cli.Vecs, V2 <: cli.Vecs}(
        f::Type{V1}, types::Type{Tuple{V2}}
    )
    return true
end
function clintrinsic(f::typeof(tuple), types::ANY)
    true
end


function clintrinsic(f::typeof(broadcast), types::ANY)
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
function Base.setindex!{T, N}(a::CLArray{T, N}, value::T, id::Integer)
    nothing
end
