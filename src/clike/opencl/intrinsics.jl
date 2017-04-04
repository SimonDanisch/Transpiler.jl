module CLIntrinsics
import ..CLTranspiler: AbstractCLIO, EmptyCLIO
using StaticArrays, Sugar
import Sugar: typename, vecname
using SpecialFunctions: erf

immutable CLArray{T, N} <: AbstractArray{T, N} end
immutable LocalMemory{T} <: AbstractArray{T, 1} end

const DeviceArray = Union{CLArray, LocalMemory}


# Number types
# Abstract types
# for now we use Int, more accurate would be Int32. But to make things simpler
# we rewrite Int to Int32 implicitely like this!
const int = Int
# same goes for float
const float = Float64
const uint = UInt
const uchar = UInt8

const ints = (int, Int32, uint, Int64)
const floats = (Float32, float)
const numbers = (ints..., floats..., Bool)

const Ints = Union{ints...}
const Floats = Union{floats...}
const Numbers = Union{numbers...}

const vector_lengths = (2, 4, 8, 16)

_vecs = []
for i in vector_lengths, T in numbers
    push!(_vecs, NTuple{i, T})
    push!(_vecs, SVector{i, T})
end

const vecs = (_vecs...)
const Vecs = Union{vecs...}
const Types = Union{vecs..., numbers..., CLArray, LocalMemory}


function typename{T, N}(io::AbstractCLIO, x::Type{CLArray{T, N}})
    if !(N in (1, 2, 3))
        # TODO, fake ND arrays with 1D array
        error("GPUArray can't have more than 3 dimensions for now")
    end
    tname = typename(io, T)
    # restrict should be fine for now, since we haven't implemented views yet!
    "__global $tname * restrict "
end
function typename{T}(io::AbstractCLIO, x::Type{LocalMemory{T}})
    tname = typename(io, T)
    "__local $tname * "
end

function fixed_array_length(T)
    N = if T <: Tuple
        length(T.parameters)
    else
        length(T)
    end
end
is_ntuple(x) = false
is_ntuple{N, T}(x::Type{NTuple{N, T}}) = true

function is_fixedsize_array{T}(::Type{T})
    (T <: SVector || is_ntuple(T)) &&
    fixed_array_length(T) in vector_lengths &&
    eltype(T) <: Numbers

end

function vecname{T}(io::AbstractCLIO, t::Type{T})
    N = fixed_array_length(T)
    return string(typename(io, eltype(T)), N)
end

@noinline function ret{T}(::Type{T})::T
    unsafe_load(Ptr{T}(C_NULL))
end

# TODO I think this needs to be UInt, but is annoying to work with!
get_global_id(dim::int) = ret(int)
get_local_id(dim::int) = ret(int)
get_group_id(dim::int) = ret(int)
get_local_size(dim::int) = ret(int)
get_global_size(dim::int) = ret(int)


const CLK_LOCAL_MEM_FENCE = Cuint(0)
barrier(::Cuint) = nothing

pow{T <: Numbers}(a::T, b::T) = ret(T)
#######################################
# globals
const functions = (
    +, -, *, /, ^, <=, .<=, !, <, >, ==, !=, |, &,
    sin, tan, sqrt, cos, mod, floor, log, atan2, max, min,
    abs, pow, log10, exp, erf
)

const Functions = Union{map(typeof, functions)...}

function clintrinsic{F <: Function, T <: Tuple}(f::F, types::Type{T})
    clintrinsic(f, Sugar.to_tuple(types))
end
function clintrinsic{F <: Function}(f::F, types::Tuple)
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
    if f == getindex && length(types) == 2 && first(types) <: DeviceArray && last(types) <: Integer
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
import .cli: clintrinsic, CLArray, DeviceArray


import Sugar.isintrinsic


function cli.clintrinsic{T}(x::Type{T})
    T <: cli.Types ||
    cli.is_fixedsize_array(T) ||
    T <: Tuple{cli.Numbers} ||
    T <: cli.uchar # uchar in ints makes 0.6 segfault -.-
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

function clintrinsic{T, I <: cli.Ints}(
        f::typeof(getindex), types::Type{Tuple{T, I}}
    )
    return is_fixedsize_array(T)
end
function clintrinsic{T <: DeviceArray, Val, I <: Integer}(
        f::typeof(setindex!), types::Type{Tuple{T, Val, I}}
    )
    return true
end

function clintrinsic(f::typeof(tuple), types::Tuple)
    true
end

function Base.getindex{T}(a::cli.LocalMemory{T}, i::Integer)
    cli.ret(T)
end
function Base.getindex{T, N}(a::CLArray{T, N}, i::Integer)
    cli.ret(T)
end
function Base.getindex{T}(a::CLArray{T, 2}, i1::Integer, i2::Integer)
    cli.ret(T)
end
function Base.getindex{T}(a::CLArray{T, 3}, i1::Integer, i2::Integer, i3::Integer)
    cli.ret(T)
end
function Base.setindex!{T}(::cli.LocalMemory{T}, ::T, ::Integer)
    nothing
end
function Base.setindex!{T, N}(a::CLArray{T, N}, value::T, i::Integer)
    nothing
end
function Base.setindex!{T}(a::CLArray{T, 2}, value::T, i1::Integer, i2::Integer)
    nothing
end
function Base.setindex!{T}(a::CLArray{T, 3}, value::T, i1::Integer, i2::Integer, i3::Integer)
    nothing
end

# TODO overload SIMD.vload, so that code can run seamlessly on the CPU as well.
for N in cli.vector_lengths
    fload = Symbol(string("vload", N))
    fstore = Symbol(string("vstore", N))
    @eval begin
        $(fload){T <: cli.Numbers, N}(i::Integer, a::CLArray{T, N}) = cli.ret(SVector{$N, T})
        function vload{T <: cli.Numbers}(::Type{SVector{$N, T}}, a::CLArray, i::Integer)
            $(fload)(i - 1, a)
        end
        clintrinsic{T <: Tuple}(f::typeof($fload), types::Type{T}) = true

        $(fstore){T <: cli.Numbers, N}(x::SVector{$N, T}, i::Integer, a::CLArray{T, N}) = nothing
        function vstore{T <: cli.Numbers}(x::SVector{$N, T}, a::CLArray, i::Integer)
            $(fstore)(x, i - 1, a)
        end
        clintrinsic{T <: Tuple}(f::typeof($fstore), types::Type{T}) = true
    end
end

# TODO Clean up this ugly mess of determining what functions not need to be compiled
# (called intrinsics here). Best would be a cl_import macro!
# Problems are, that they either need to define a function stub for Inference
# or just leave them if already defined in base, but still add the information
# We can solve this by having two macros. cl_pirate for functions in base
# and cl_import for new functions
macro cl_pirate(func)
end
macro cl_import(func)

end
