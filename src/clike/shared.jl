using Sugar
import Sugar: ASTIO, LazyMethod, typename, functionname, _typename, show_name
import Sugar: supports_overloading, show_type, show_function

abstract CIO <: ASTIO

global replace_unsupported, empty_replace_cache!

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
    !(char in ('.', '#'))  # some ascii codes are not allowed
end

function symbol_hygiene(io::CIO, sym)
    # TODO figure out what other things are not allowed
    # TODO startswith gl_, but allow variables that are actually valid inbuilds
    res_io = IOBuffer()
    for (i, char) in enumerate(string(sym))
        res = if is_supported_char(io, char)
            print(res_io, char)
        else
            if i == 1 # can't start with number
                print(res_io, '_')
            end
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
    elseif (N in (2, 3, 4, 8)) && T <: Number # TODO look up numbers again!
        Sugar.vecname(io, t)
    else
        string(typename(io, T), '[', N, ']')
    end
end
# don't do hygiene
typename{N, T}(io::CIO, t::Type{SVector{N, T}}) = _typename(io, t)
function _typename{N, T}(io::CIO, t::Type{SVector{N, T}})
    if N == 1 && T <: Number
        return typename(io, T)
    elseif (N in (2, 3, 4, 8)) && T <: Number # TODO look up numbers again!
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

function functionname(io::CIO, f, types)
    if isa(f, Type) || isa(f, Expr)
        # This should only happen, if the function is actually a type
        if isa(f, Expr)
            f = f.typ
        end
        return string('(', _typename(io, f), ')')
    end
    method = try
        LazyMethod(io.method, f, types)
    catch e
        error("Couldn't create function $f with $types")
    end
    f_sym = Symbol(typeof(f).name.mt.name)
    if Sugar.isintrinsic(method)
        return f_sym # intrinsic operators don't need hygiene!
    end
    str = if supports_overloading(io)
        string(f_sym)
    else
        string(f_sym, '_', signature_hash(types))
    end
    symbol_hygiene(io, str)
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
function show_function(io::CIO, f, types)
    print(io, functionname(io, f, types))
end

function Base.show_unquoted(io::CIO, slot::Slot, ::Int, ::Int)
    show_name(io, slot)
end
