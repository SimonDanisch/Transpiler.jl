module CLIntrinsics

import ..Transpiler: AbstractCLIO, EmptyCLIO
import ..Transpiler: ints, floats, numbers, Numbers, Floats, int, Ints, uchar
import ..Transpiler: fixed_array_length, is_fixedsize_array
import ..Transpiler: ret, vecs, Vecs, vector_lengths, functions


using StaticArrays, Sugar
import Sugar: typename, vecname
using SpecialFunctions: erf

immutable CLArray{T, N} <: AbstractArray{T, N} end
immutable LocalMemory{T} <: AbstractArray{T, 1} end

const CLDeviceArray = Union{CLArray, LocalMemory}
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

function vecname{T}(io::AbstractCLIO, t::Type{T})
    N = fixed_array_length(T)
    return string(typename(io, eltype(T)), N)
end


# TODO I think this needs to be UInt, but is annoying to work with!
get_global_id(dim::Integer) = ret(Cuint)
get_local_id(dim::Integer) = ret(Cuint)
get_group_id(dim::Integer) = ret(Cuint)
get_local_size(dim::Integer) = ret(Cuint)
get_global_size(dim::Integer) = ret(Cuint)

select(a::T, b::T, c::Bool) where T = ret(T)


const CLK_LOCAL_MEM_FENCE = Cuint(0)
const CLK_GLOBAL_MEM_FENCE = Cuint(0)
barrier(::Cuint) = nothing
mem_fence(::Cuint) = nothing
#######################################
# globals

const Functions = Union{map(typeof, (functions..., erf, erfc))..., }

function clintrinsic{F <: Function, T <: Tuple}(f::F, types::Type{T})
    clintrinsic(f, Sugar.to_tuple(types))
end
function is_cl_native_type(T)
    T <: Types || is_fixedsize_array(T) || T <: Tuple{T} where T <: Numbers
end
function clintrinsic{F <: Function}(f::F, types::Tuple)
    if f == broadcast
        BF = types[1]
        if BF <: Functions && all(T-> T <: Types, types[2:end])
            return true
        end
    end
    if f == getindex && length(types) == 2 && first(types) <: CLDeviceArray && last(types) <: Integer
        return !is_fixedsize_array(eltype(first(types)))
    end
    (F <: Functions && all(is_cl_native_type, types)) && return true
    m = methods(f)
    isempty(m) && return false
    sym = first(m).name
    (
        # if any intrinsic funtion stub matches
        isdefined(CLIntrinsics, sym) &&
        Base.binding_module(CLIntrinsics, sym) == CLIntrinsics &&
        length(methods(f, types)) == 1
    )
end

end # end CLIntrinsics


using .CLIntrinsics

const cli = CLIntrinsics
import .cli: clintrinsic, CLArray, CLDeviceArray


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
        cli.clintrinsic(x.signature[1], Sugar.to_tuple(x.signature[2])) ||
        cli.clintrinsic(x.signature[1], x.signature[2])
    else
        cli.clintrinsic(x.signature)
    end
end

# copied from rewriting. TODO share implementation!

# Make constructors inbuild for now. TODO, only make default constructors inbuild
function clintrinsic{T}(f::Type{T}, types::ANY)
    return true
end

function clintrinsic{T, I <: Integer}(
        f::typeof(getindex), types::Type{Tuple{T, I}}
    )
    return is_fixedsize_array(T)
end
function clintrinsic{T <: CLDeviceArray, Val, I <: Integer}(
        f::typeof(setindex!), types::Type{Tuple{T, Val, I}}
    )
    return !is_fixedsize_array(Val) # fixed size array setindex is no intrinsic, since it uses vstore
end


function clintrinsic(f::typeof(tuple), types::Tuple)
    true
end

Base.getindex{T}(a::cli.LocalMemory{T}, i::Integer) = cli.ret(T)
Base.getindex{T, N}(a::CLArray{T, N}, i::Integer) = cli.ret(T)

Base.setindex!{T}(::cli.LocalMemory{T}, ::T, ::Integer) = nothing
Base.setindex!{T, N}(a::CLArray{T, N}, value::T, i::Integer) = nothing


# TODO overload SIMD.vload, so that code can run seamlessly on the CPU as well.
for VecType in (NTuple, SVector)
    for N in cli.vector_lengths
        fload = Symbol(string("vload", N))
        fstore = Symbol(string("vstore", N))
        VType = :($VecType{$N, T})
        if VecType == NTuple
            @eval begin
                $(fload){T <: cli.Numbers, N}(i::Integer, a::CLArray{T, N}) = cli.ret(NTuple{$N, T})
                $(fstore){T <: cli.Numbers, N}(x::$VType, i::Integer, a::CLArray{T, N}) = nothing
                function cli.clintrinsic(f::typeof($fstore), types::Tuple)
                    length(types) == 3 || return false
                    is_fixedsize_array(types[1]) && types[2] <: Integer && types[3] <: CLArray
                end
                function cli.clintrinsic(f::typeof($fload), types::Tuple)
                    length(types) == 2 || return false
                    types[1] <: Integer && types[2] <: CLArray
                end
            end
        end
        @eval begin
            function vload{T <: cli.Numbers, N}(::Type{$VType}, a::CLArray{$VType, N}, i::Integer)
                $VType($(fload)(i - 1, CLArray{T, N}(a)))
            end
            function vstore{T <: cli.Numbers, N}(x::$VType, a::CLArray{$VType, N}, i::Integer)
                $(fstore)(x, i - 1, CLArray{T, N}(a))
            end
        end
    end
end

Base.getindex{T <: Vecs, N}(a::CLArray{T, N}, i::Integer) = vload(T, a, i)
function Base.setindex!{T <: Vecs}(a::CLArray{T}, value::T, i::Integer)
    vstore(value, a, i)
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

function gpu_ind2sub{T}(dims, ind::T)
    Base.@_inline_meta
    _ind2sub(dims, ind - T(1))
end

_ind2sub{T}(::Tuple{}, ind::T) = (ind + T(1),)
function _ind2sub{T}(indslast::NTuple{1}, ind::T)
    Base.@_inline_meta
    ((ind + T(1)),)
end
function _ind2sub{T}(inds, ind::T)
    Base.@_inline_meta
    r1 = inds[1]
    indnext = div(ind, r1)
    f = T(1); l = r1
    (ind-l*indnext+f, _ind2sub(Base.tail(inds), indnext)...)
end
