module CLIntrinsics

using MacroTools

import ..Transpiler: AbstractCLIO, EmptyCLIO
import ..Transpiler: ints, floats, numbers, Numbers, Floats, int, Ints, uchar
import ..Transpiler: fixed_array_length, is_fixedsize_array
import ..Transpiler: ret, vecs, Vecs, vector_lengths, functions

using StaticArrays, Sugar
import Sugar: typename, vecname
using SpecialFunctions: erf, erfc

# TODO, these are rather global pointers and this should be represented in the type
immutable CLArray{T, N} <: AbstractArray{T, N} end
immutable LocalMemory{T} <: AbstractArray{T, 1} end

const CLDeviceArray = Union{CLArray, LocalMemory}
const Types = Union{vecs..., numbers..., CLArray, LocalMemory}

#########
# GLOBALS
const CLK_LOCAL_MEM_FENCE = Cuint(0)
const CLK_GLOBAL_MEM_FENCE = Cuint(0)

const intrinsic_signatures = Dict{Function, Any}()

macro cl_intrinsic(expr)
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

@cl_intrinsic get_global_id(dim::Integer) = ret(Cuint)
@cl_intrinsic get_local_id(::Integer) = ret(Cuint)
@cl_intrinsic get_group_id(::Integer) = ret(Cuint)
@cl_intrinsic get_local_size(::Integer) = ret(Cuint)
@cl_intrinsic get_global_size(::Integer) = ret(Cuint)
@cl_intrinsic select(::T, ::T, ::Bool) where {T} = ret(T)

@cl_intrinsic barrier(::Cuint) = nothing
@cl_intrinsic mem_fence(::Cuint) = nothing

@cl_intrinsic erfc(::T) where T <: Floats = ret(T)
@cl_intrinsic erf(::T) where T <: Floats = ret(T)


for N in vector_lengths
    fload = Symbol(string("vload", N))
    fstore = Symbol(string("vstore", N))
    @eval begin
        @cl_intrinsic $(fload)(i::Integer, a::CLArray{T, N}) where {T <: Numbers, N} = ret(NTuple{$N, T})
        @cl_intrinsic $(fstore)(x::NTuple{$N, T}, i::Integer, a::CLArray{T, N}) where {T <: Numbers, N} = nothing
    end
end


end # end CLIntrinsics


using .CLIntrinsics

const cli = CLIntrinsics
import .cli: CLArray, CLDeviceArray

import Sugar: typename, isintrinsic

function is_native_type(m::CLMethod, T)
    T <: cli.Types || is_fixedsize_array(m, T) || T <: Tuple{T} where T <: cli.Numbers
end

function isintrinsic(m::CLMethod, func::ANY, sig_tuple::ANY)
    # constructors are intrinsic. TODO more thorow lookup to match actual inbuild constructor
    isa(func, DataType) && return true
    func == tuple && return true # TODO match against all Base intrinsics?
    func == getfield && sig_tuple <: (Tuple{X, Symbol} where X) && return true
    func == getfield && sig_tuple <: (Tuple{X, Integer} where X <: Tuple) && return true
    # shared intrinsic functions should all work on all native types.
    # TODO, find exceptions where this isn't true
    func in functions && all(x-> is_native_type(m, x), Sugar.to_tuple(sig_tuple)) && return true
    haskey(cli.intrinsic_signatures, func) || return false
    sig = cli.intrinsic_signatures[func]
    sig_tuple <: sig
end

function isintrinsic(x::CLMethod)
    if isfunction(x)
        isintrinsic(x, x.signature...)
    else
        is_native_type(x, x.signature)
    end
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
        @eval begin
            function vload{T <: cli.Numbers, N, IT <: Integer}(::Type{$VType}, a::CLArray{$VType, N}, i::IT)
                $VType(cli.$(fload)(i - IT(1), CLArray{T, N}(a)))
            end
            function vstore{T <: cli.Numbers, N, IT <: Integer}(x::$VType, a::CLArray{$VType, N}, i::IT)
                cli.$(fstore)(Tuple(x), i - IT(1), CLArray{T, N}(a))
            end
        end
    end
end

Base.getindex{T <: Vecs, N}(a::CLArray{T, N}, i::Integer) = vload(T, a, i)
function Base.setindex!{T <: Vecs}(a::CLArray{T}, value::T, i::Integer)
    vstore(value, a, i)
end

supports_indexing(m::LazyMethod, ::Type{<: CLDeviceArray}) = true

function supports_indices(m::LazyMethod, ::Type{ <: CLArray{T, N}}, index_types) where {T, N}
    is_fixedsize_array(m, T) && return false # fixed size arrays are implemented via vstore/load
    length(index_types) == 1 && index_types[1] <: Integer
end
function supports_indices(m::LazyMethod, ::Type{ <: CLDeviceArray}, index_types)
    length(index_types) == 1 && index_types[1] <: Integer
end
function supports_indices(m::LazyMethod, ::Type{<: Tuple}, index_types)
    length(index_types) == 1 && index_types[1] <: Integer
end


function typename{T, N}(io::AbstractCLIO, x::Type{CLArray{T, N}})
    tname = typename(io, T)
    # restrict should be fine for now, since we haven't implemented views yet!
    "__global $tname * restrict "
end
function typename{T}(io::AbstractCLIO, x::Type{cli.LocalMemory{T}})
    tname = typename(io, T)
    "__local $tname * "
end

function Sugar.vecname(io::AbstractCLIO, t::Type{T}) where T
    N = fixed_array_length(T)
    return string(typename(io, eltype(T)), N)
end
