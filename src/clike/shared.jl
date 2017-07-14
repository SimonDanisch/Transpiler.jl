import Sugar: ASTIO, LazyMethod, typename, functionname, _typename, show_name
import Sugar: supports_overloading, show_type, show_function
import SpecialFunctions: erf, erfc

@compat abstract type CIO <: ASTIO end

immutable EmptyStruct
    # Emtpy structs are not supported in OpenCL, which is why we emit a struct
    # with one floating point field
    x::Float32
    EmptyStruct() = new()
end

@noinline function ret{T}(::Type{T})::T
    unsafe_load(Ptr{T}(C_NULL))
end
# Number types
# Abstract types
# for now we use Int, more accurate would be Int32. But to make things simpler
# we rewrite Int to Int32 implicitely like this!
const int = Int32
# same goes for float
const float = Float64
const uint = UInt32
const uchar = UInt8

const ints = (int, Int32, uint, Int64)
const floats = (Float32, float)
const numbers = (ints..., floats..., Bool)

const Ints = Union{ints...}
const Floats = Union{floats...}
const Numbers = Union{numbers...}

const vector_lengths = (2, 3, 4, 8, 16)
_vecs = []
for i in vector_lengths, T in numbers
    push!(_vecs, NTuple{i, T})
    push!(_vecs, SVector{i, T})
end
const vecs = (_vecs...)
const Vecs = Union{vecs...}

pow{T <: Numbers}(a::T, b::T) = a ^ b


"""
smoothstep performs smooth Hermite interpolation between 0 and 1 when edge0 < x < edge1. This is useful in cases where a threshold function with a smooth transition is desired. smoothstep is equivalent to:
```
    t = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
```
Results are undefined if edge0 ≥ edge1.
"""
function smoothstep{T}(edge0, edge1, x::T)
    t = clamp.((x .- edge0) ./ (edge1 .- edge0), T(0), T(1))
    return t * t * (T(3) - T(2) * t)
end

"""
mix performs a linear interpolation between x and y using a to weight between them. The return value is computed as
`x .* (T(1) .- a) .+ y .* a`
"""
mix{T}(x, y, a::T) = x .* (T(1) .- a) .+ y .* a

fract(x) = x - floor(x)
fabs(x::AbstractFloat) = abs(x)

#######################################
# globals
const functions = (
    +, -, *, /, ^, <=, .<=, !, <, >, ==, !=, |, &,
    sin, tan, sqrt, cos, mod, round, floor, fract, log, atan2, atan, max, min,
    abs, pow, log10, exp, normalize, cross, dot, smoothstep, mix, norm,
    length, clamp, cospi, sinpi, asin, fma, fabs
)

global replace_unsupported, empty_replace_cache!
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
    (T <: StaticVector || is_ntuple(T)) &&
    fixed_array_length(T) in vector_lengths &&
    eltype(T) <: Numbers
end


let _unsupported_id = 0
    const unsupported_replace_dict = Dict{Char, String}()
    function empty_replace_cache!()
        empty!(unsupported_replace_dict)
        return
    end
    """
    Creates a unique replacement for some character
    """
    function replace_unsupported(char::Char)
        get!(unsupported_replace_dict, char) do
            _unsupported_id += 1
            string(_unsupported_id)
        end
    end
end

function is_supported_char(io::CIO, char)
    # Lets just assume for simplicity, that only ascii non operators are supported
    # in a name
    isascii(char) &&
    !Base.isoperator(Symbol(char)) &&
    !(char in ('.', '#', '(', ')', ','))  # some ascii codes are not allowed
end

function symbol_hygiene(io::CIO, sym)
    # TODO figure out what other things are not allowed
    # TODO startswith gl_, but allow variables that are actually valid inbuilds
    res_io = IOBuffer()
    for (i, char) in enumerate(string(sym))
        res = if is_supported_char(io, char)
            print(res_io, char)
        else
            i == 1 && print(res_io, 'x') # can't start with number
            print(res_io, replace_unsupported(char)) # get a
        end
    end
    String(take!(res_io))
end

typename(io::CIO, x) = Symbol(symbol_hygiene(io, _typename(io, x)))

# don't do hygiene
typename{N, T}(io::CIO, t::Type{NTuple{N, T}}) = _typename(io, t)
function _typename{N, T}(io::CIO, t::Type{NTuple{N, T}})
    # Rewrite rewrite ntuples as glsl arrays
    # e.g. float[3]
    if N == 1 && T <: Number
        return typename(io, T)
    elseif (N in vector_lengths) && T <: Number # TODO look up numbers again!
        Sugar.vecname(io, t)
    else
        string("__constant ", typename(io, T), " * ")
    end
end

const vector_lengths = (2, 3, 4, 8, 16)
# don't do hygiene
typename{SV <: StaticVector}(io::CIO, t::Type{SV}) = _typename(io, t)
function _typename{SV <: StaticVector}(io::CIO, t::Type{SV})
    N, T = length(SV), eltype(SV)
    if N == 1 && T <: Number
        return typename(io, T)
    elseif (N in vector_lengths) && T <: Number
        Sugar.vecname(io, t)
    else
        string(typename(io, T), '[', N, ']')
    end
end

_typename(io::CIO, T::QuoteNode) = _typename(io, T.value)

function _typename(io::CIO, T)
    str = if isa(T, Expr) && T.head == :curly
        string(T, "_", join(T.args, "_"))
    elseif isa(T, Symbol)
        string(T)
    elseif isa(T, Tuple)
        str = "Tuple_"
        if !isempty(t.parameters)
            tstr = map(x-> typename(io, x), t.parameters)
            str *= join(tstr, "_")
        end
        str
    elseif isa(T, Type)
        str = string(T.name.name)
        if !isempty(T.parameters)
            tstr = map(T.parameters) do t
                if isa(t, DataType)
                    typename(io, t)
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



_typename(io::CIO, x::Union{AbstractString, Symbol}) = x

if VERSION < v"0.6"
    _typename(io::CIO, ::typeof(.+)) = "+"
    _typename(io::CIO, ::typeof(.-)) = "-"
    _typename(io::CIO, ::typeof(.*)) = "*"
    _typename(io::CIO, ::typeof(./)) = "/"
end

_typename{T <: Number}(io::CIO, x::Type{Tuple{T}}) = _typename(io, T)
_typename(io::CIO, x::Type{Void}) = "void"
_typename(io::CIO, x::Type{Float64}) = "float"
_typename(io::CIO, x::Type{Float32}) = "float"
_typename(io::CIO, x::Type{Int}) = "int"
_typename(io::CIO, x::Type{Int32}) = "int"
_typename(io::CIO, x::Type{UInt}) = "uint"
_typename(io::CIO, x::Type{UInt8}) = "uchar"
_typename(io::CIO, x::Type{Bool}) = "bool"
_typename{T}(io::CIO, x::Type{Ptr{T}}) = "$(typename(io, T)) *"

# TODO this will be annoying on 0.6
# _typename(x::typeof(cli.:(*))) = "*"
# _typename(x::typeof(cli.:(<=))) = "lessThanEqual"
# _typename(x::typeof(cli.:(+))) = "+"

function _typename{F <: Function}(io::CIO, f::F)
    _typename(io, F.name.mt.name)
end
function _typename{F <: Function}(io::CIO, f::Type{F})
    string(F)
end

global signature_hash
let hash_dict = Dict{Any, Int}(), counter = 0
    """
    Returns a unique ID for a type signature, which is as small as possible!
    """
    function signature_hash(types)
        get!(hash_dict, Sugar.to_tuple(types)) do
            counter += 1
            counter
        end
    end
end

function functionname(io::IO, method::LazyMethod)
    if istype(method)
        # This should only happen, if the function is actually a type
        return string('(', _typename(io, method.signature), ')')
    end
    f_sym = Symbol(typeof(Sugar.getfunction(method)).name.mt.name)
    if Sugar.isintrinsic(method)
        return f_sym # intrinsic operators don't need hygiene!
    end
    str = if isa(io, Sugar.ASTIO) && supports_overloading(io)
        string(f_sym)
    else
        string(f_sym, '_', signature_hash(method.signature[2]))
    end
    if isa(io, Sugar.ASTIO)
        symbol_hygiene(io, str)
    else
        str
    end
end

function show_name(io::CIO, x)
    print(io, symbol_hygiene(io, x))
end
function show_name(io::CIO, x::Union{Slot, SSAValue})
    name = Sugar.slotname(io.method, x)
    show_name(io, name)
end

function show_type(io::CIO, x)
    print(io, typename(io, x))
end

function Base.show_unquoted(io::CIO, slot::Slot, ::Int, ::Int)
    show_name(io, slot)
end

function c_fieldname(T, i)
    name = try
        Base.fieldname(T, i)
    catch e
        error("couldn't get field name for $T")
    end
    if isa(name, Integer) # for types without fieldnames (Tuple)
        "field$name"
    else
        symbol_hygiene(EmptyGLIO(), name)
    end
end

function typed_type_fields(T)
    nf = nfields(T)
    fields = []
    if nf == 0 # structs can't be empty
        # we use bool as a short placeholder type.
        # TODO, are there cases where bool is no good?
        push!(fields, :(emtpy::Float32))
    else
        for i in 1:nf
            FT = fieldtype(T, i)
            tname = Symbol(typename(EmptyGLIO(), FT))
            fname = Symbol(c_fieldname(T, i))
            push!(fields, :($fname::$tname))
        end
    end
    fields
end
