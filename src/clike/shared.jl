#=
This file contains functions taken and modified from Julia/base/show.jl
license of show.jl:
Copyright (c) 2009-2016: Jeff Bezanson, Stefan Karpinski, Viral B. Shah, and other contributors:
https://github.com/JuliaLang/julia/contributors
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
=#

import Base: indent_width, uni_ops, expr_infix_wide
import Base: all_ops, expr_calls, show_block
import Base: show_list, show_enclosed_list, operator_precedence
import Base: is_quoted, is_expr, expr_infix_any
import Base: show_call, show_unquoted, show

import Sugar: ASTIO, LazyMethod, typename, functionname, _typename, show_name
import Sugar: supports_overloading, show_type, show_function, is_native_type, isintrinsic

@compat abstract type CIO <: ASTIO end
immutable EmptyCIO <: CIO
end

immutable EmptyStruct
    # Emtpy structs are not supported in OpenCL, which is why we emit a struct
    # with one floating point field
    x::Int32
    EmptyStruct() = new()
end

# helper function to fake a return type to type inference for intrinsic function stabs
@noinline function ret{T}(::Type{T})::T
    unsafe_load(Ptr{T}(C_NULL))
end
# Number types
# Abstract types
# for now we use Int, more accurate would be Int32. But to make things simpler
# we rewrite Int to Int32 implicitely like this!
const int = Int32
# same goes for float
const float = Float32
const uint = UInt32
const uchar = UInt8

const ints = (UInt64, UInt32, UInt8, Int64, Int32, Int8)
const floats = (Float32, Float64)
const numbers = (ints..., floats..., Bool)

const Ints = Union{ints...}
const Floats = Union{floats...}
const Numbers = Union{numbers...}

const vector_lengths = (2, 3, 4, 8, 16)
# Maybe we should remove this and leave it to `is_fixedsize_array`, which is extensible
_vecs = []
for i in vector_lengths, T in numbers
    T == Bool && continue
    push!(_vecs, NTuple{i, T})
    push!(_vecs, SVector{i, T})
end
const vecs = (_vecs...)
const Vecs = Union{vecs...}


##################################################
# Intrinsics that don't have a julia counterpart:

pow{T <: Numbers}(a::T, b::T) = a ^ b
cl_pow{T1, T2}(a::T1, b::T2) = pow(promote(a, b)...)
cl_pow{T <: Numbers}(a::T, b::T) = pow(a, b)

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
# shared common intrinsic functions

const functions = (
    +, -, *, /, ^, <=, .<=, !, <, >, ==, !=, |, &, %,
    <<, >>,
    sqrt, fract, log,
    round, floor, ceil, trunc,
    sin, sinpi, sinh, asin, asinh,
    cos, cospi, cosh, acos, acosh,
    tan, tanh, atan, atanh, atan2,# atanpi, atan2pi, <- julia doesnt have those?!
    max, min,
    abs, pow, normalize, cross, dot, smoothstep, mix,
    exp2, exp10, expm1,
    log, log2, log10, log1p,
    length, clamp, fma, fabs, isinf, isnan, sign,
    cbrt, copysign, signbit
    # fast math
)

function fixed_array_length(T)
    N = if T <: Tuple
        length(T.parameters)
    else
        length(T)
    end
end
is_ntuple(x) = false
is_ntuple{N, T}(x::Type{NTuple{N, T}}) = true


const vector_lengths = (2, 3, 4, 8, 16)

# This IO thing gets annoying! TODO, remove IO as dispatch object and use exclusively LazyMethods

is_fixedsize_array{T}(::Type{T}) = is_fixedsize_array(nothing, T)
function is_fixedsize_array{T}(m, ::Type{T})
    (T <: StaticVector || is_ntuple(T)) &&
    fixed_array_length(T) in vector_lengths &&
    eltype(T) <: Numbers &&
    eltype(T) != Bool
end

# Functionality to remove unsupported characters from the source
global replace_unsupported, empty_replace_cache!
let _unsupported_id = 0
    const unsupported_replace_dict = Dict{Char, String}()
    function empty_replace_cache!()
        _unsupported_id = 0
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

is_supported_char(io::IO, char) = true
function is_supported_char(io::CIO, char)
    # Lets just assume for simplicity, that only ascii non operators are supported
    # in a name
    isascii(char) &&
    !Base.isoperator(Symbol(char)) &&
    !(char in ('.', '#', '(', ')', ',', '{', '}', ' '))  # some ascii codes are not allowed
end

"""
Makes sure a symbol is well formed in target source
"""
function symbol_hygiene(io::IO, sym)
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



_typename(io::IO, T::QuoteNode) = _typename(io, T.value)
julia_name(x::Type{Type{T}}) where T = string(T)

function _typename(io::IO, x)
    str = if isa(x, Expr) && x.head == :curly
        string(x, "_", join(x.args, "_"))
    elseif isa(x, Symbol)
        string(x)
    elseif isa(x, UnionAll)
        string(x)
    elseif isa(x, DataType)
        T = x
        if T <: Tuple{X} where X <: Numbers
            # We map Tuple{1, X} to the intrinsic types of it for better performance (usually this would need to be a struct)
            # TODO benchmark if this terrible idea is worth it
            typename(io, eltype(T))
        elseif T <: Type
            # make names unique when it was a type of Type{X}
            string(T)
        elseif is_fixedsize_array(io, T)
            Sugar.vecname(io, T)
        elseif T <: Tuple
            str = if isempty(T.parameters)
                "EmptyTuple_"
            elseif T == Tuple
                "EmptyTuple"
            else
                str = "Tuple_"
                tstr = map(T.parameters) do t
                    if t <: Tuple{X} where X
                        string("Tupl_", typename(io, t))
                    else
                        typename(io, t)
                    end
                end
                str *= join(tstr, "_")
            end
            str
        else
            str = string(T.name.name)
            if !isempty(T.parameters)
                tstr = map(T.parameters) do t
                    if isa(t, DataType)
                        if t <: Tuple{X} where X
                            string("Tupl_", typename(io, t))
                        else
                            typename(io, t)
                        end
                    else
                        string(t)
                    end
                end
                str *= string("_", join(tstr, "_"))
            end
            str
        end
    else
        error("Not transpilable: $x")
    end
    return str
end

_typename(io::CIO, x::Union{AbstractString, Symbol}) = x

_typename{T <: Numbers}(io::IO, x::Type{Tuple{T}}) = _typename(io, T)
_typename(io::IO, x::Type{Float64}) = "double"
_typename(io::IO, x::Type{Float32}) = "float"
_typename(io::IO, x::Type{Int64}) = "long"
_typename(io::IO, x::Type{Int32}) = "int"
_typename(io::IO, x::Type{UInt32}) = "uint"
_typename(io::IO, x::Type{UInt64}) = "ulong"
_typename(io::IO, x::Type{UInt8}) = "uchar"
_typename(io::IO, x::Type{Bool}) = "JLBool"
_typename{T}(io::IO, x::Type{Ptr{T}}) = "$(typename(io, T)) *"

_typename{F <: Function}(io::CIO, f::F) = _typename(io, F.name.mt.name)
_typename{F <: Function}(io::CIO, f::Type{F}) = string(F)

global signature_hash, empty_hash_dict!
let hash_dict = Dict{Any, Int}(), counter = 0
    function empty_hash_dict!()
        empty!(hash_dict)
        counter = 0
        return
    end
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

function Sugar.functionname(io::CIO, method::Type{T}) where T
    string('(', typename(io, T), ')')
end
function Sugar.functionname(io::CIO, method::LazyMethod)
    func = if istype(method)
        # This should only happen, if the function is actually a type constructor
        method.signature
    else
        Sugar.getfunction(method)
    end
    if func == %
        # % seems to be an operator that is printed as rem (?!)
        # TODO, are there more? This is only important for operators that are also intrinsics
        return :(%)
    end
    if func == cli.intrinsic_select
        return :select
    end
    f_sym = if isa(func, Type)
        functionname(io, func)
    else
        Base.function_name(func)
    end
    if Sugar.isintrinsic(method)
        return f_sym # intrinsic operators don't need hygiene!
    end
    str = if isfunction(method) && !supports_overloading(io)
        string(f_sym, '_', signature_hash(method.signature[2]))
    elseif istype(method)
        return string(f_sym)
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

function c_fieldname(m::LazyMethod, T, i::Integer)
    str = if isleaftype(T)
        name = if is_fixedsize_array(m, T)
            fixed_size_array_fieldname(m, T, i)
        else
            Base.fieldname(T, i)
        end
        if isa(name, Integer) # for types without fieldnames (Tuple)
            "field$name"
        else
            symbol_hygiene(EmptyCIO(), name)
        end
    else
        error("Found abstract type: $T")
    end
    Symbol(str)
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
            tname = typename(EmptyCIO(), FT)
            fname = c_fieldname(T, i)
            push!(fields, :($fname::$tname))
        end
    end
    fields
end
Sugar.show_comment(io::CIO, comment) = println(io, "// ", comment)

show_linenumber(io::CIO, line) = show_comment(io, " line $line:")
show_linenumber(io::CIO, line, file) = show_comment(io, "$file $line $line")


function show_unquoted(io::CIO, x::Type, ::Int, ::Int)
    print(io, "TYP_INST_", typename(io, Type{x}))
end
function show_unquoted(io::CIO, sym::Symbol, ::Int, ::Int)
    print(io, Symbol(symbol_hygiene(io, sym)))
end
function show_unquoted(io::CIO, ssa::SSAValue, ::Int, ::Int)
    print(io, Sugar.ssavalue_name(ssa))
end
function show_unquoted(io::CIO, f::F, ::Int, ::Int) where F <: Sugar.AllFuncs
    print(io, "FUNC_INST_", typename(io, F))
end

const expr_calls_extended = merge(expr_calls, Dict(:constructor => ('{', '}')))
# show a normal (non-operator) function call, e.g. f(x,y) or A[z]
function show_call(io::CIO, head, func, func_args, indent)
    if head == :curly && Sugar.expr_type(func) <: Type# typeconstructors
        print(io, "TYP_INST_") # use the global constant instance of the type
        show_unquoted(io, func, indent)
        return
    end
    op, cl = expr_calls_extended[head]
    # print(io, '(')
    if head == :ref
        FT = Sugar.expr_type(func)
        if is_fixedsize_array(FT) && length(func_args) == 1
            # Special case for small vectors an an integer variable as index
            IDXT = Sugar.expr_type(func_args[1])
            if IDXT <: Integer && !isa(func_args[1], Integer)
                ET = eltype(FT)
                print(io, "(($(typename(io, ET))*)&") # convert to pointer
                show_unquoted(io, func, indent)
                print(io, ")[")
                show_unquoted(io, func_args[1])
                print(io, "]")
                return
            end
        end
        print(io, '(')
        show_unquoted(io, func, indent)
        print(io, ')')
        if FT <: Tuple{T} where T <: cli.Numbers
            # we Tuple{<: Numbers} is treated as scalar, so we don't print the index expression
            return
        end
    else
        print(io, functionname(io, func))
    end
    if head == :(.)
        print(io, '.')
    end
    addition_tracked_object = []
    m = io.method
    for elem in func_args
        T = Sugar.expr_type(m, elem)
        contains, fields = Sugar.contains_tracked_type(m, T)
        if contains && haskey(m.cache, :tracked_types)
            fields = m.cache[:tracked_types][elem]
            for field in fields
                push!(addition_tracked_object, last(field))
            end
        end
    end
    show_enclosed_list(io, op, vcat(func_args, addition_tracked_object), ", ", cl, indent)
    return
end

# show a block, e g if/for/etc
function show_block(io::CIO, head, args::Vector, body, indent::Int)
    if isempty(args) && isa(body, Expr) && isempty(body.args)
        return # everything empty, let's not make a fool of ourselves and print something
    end
    if isempty(args)
        print(io, head, '{')
    else
        print(io, head, '(')
        show_list(io, args, ", ", indent)
        print(io, "){")
    end
    ind = head === :module || head === :baremodule ? indent : indent + indent_width
    exs = (is_expr(body, :block) || is_expr(body, :body)) ? body.args : Any[body]
    for (i, ex) in enumerate(exs)
        sep = i == 1 ? "" : ";"
        print(io, sep, '\n', " "^ind)
        show_unquoted(io, ex, ind, -1)
    end
    print(io, ";\n", " "^indent)
end


function show_unquoted(io::CIO, ex::Expr, indent::Int, prec::Int)
    line_number = 0 # TODO include line numbers
    head, args, nargs = ex.head, ex.args, length(ex.args)
    # dot (i.e. "x.y"), but not compact broadcast exps
    if head == :(.) && !is_expr(args[2], :tuple)
        show_unquoted(io, args[1], indent + indent_width)
        print(io, '.')
        if is_quoted(args[2])
            show_unquoted(io, unquoted(args[2]), indent + indent_width)
        else
            print(io, '(')
            show_unquoted(io, args[2], indent + indent_width)
            print(io, ')')
        end

    # variable declaration TODO might this occure in normal code?
    elseif head == :(::) && nargs == 2
        show_type(io, args[2])
        print(io, ' ')
        show_name(io, args[1])
    # infix (i.e. "x<:y" or "x = y")

    elseif (head in expr_infix_any && (nargs == 2) || (head == :(:)) && nargs == 3)
        func_prec = operator_precedence(head)
        head_ = head in expr_infix_wide ? " $head " : head
        if func_prec <= prec
            show_enclosed_list(io, '(', args, head_, ')', indent, func_prec, true)
        else
            show_list(io, args, head_, indent, func_prec, true)
        end

    # function call
    elseif head === :call && nargs >= 1
        f = first(args)
        func_args = args[2:end]
        fname = Symbol(functionname(io, f))
        if fname == :getfield && nargs == 3
            accessed, fieldname = args[2], args[3]
            m = io.method
            tt = Sugar.expr_type(ex)
            show_unquoted(io, accessed, indent) # type to be accessed
            print(io, '.')
            fieldname = isa(fieldname, QuoteNode) ? fieldname.value : fieldname
            print(io, fieldname)
        else
            func_prec = operator_precedence(fname)
            # scalar multiplication (i.e. "100x")
            if (
                    fname === :* &&
                    length(func_args) == 2 && isa(func_args[1], Real) &&
                    isa(func_args[2], Symbol)
                )
                if func_prec <= prec
                    show_enclosed_list(io, '(', func_args, "", ')', indent, func_prec)
                else
                    show_list(io, func_args, "", indent, func_prec)
                end

            # unary operator (i.e. "!z")
            elseif isa(fname, Symbol) && fname in uni_ops && length(func_args) == 1
                print(io, fname)
                if isa(func_args[1], Expr) || func_args[1] in all_ops
                    show_enclosed_list(io, '(', func_args, ",", ')', indent, func_prec)
                else
                    show_unquoted(io, func_args[1])
                end

            # binary operator (i.e. "x + y")
            elseif func_prec > 0 # is a binary operator
                na = length(func_args)
                if (na == 2 || (na > 2 && fname in (:+, :++, :*))) && all(!isa(a, Expr) || a.head !== :... for a in func_args)
                    sep = " $fname "
                    if func_prec <= prec
                        show_enclosed_list(io, '(', func_args, sep, ')', indent, func_prec, true)
                    else
                        show_list(io, func_args, sep, indent, func_prec, true)
                    end
                elseif na == 1
                    # 1-argument call to normally-binary operator
                    op, cl = expr_calls[head]
                    show_unquoted(io, f, indent)
                    show_enclosed_list(io, op, func_args, ",", cl, indent)
                else
                    show_call(io, head, f, func_args, indent)
                end

            # normal function (i.e. "f(x,y)")
            else
                show_call(io, head, f, func_args, indent)
            end
        end
    # other call-like expressions ("A[1,2]", "T{X,Y}", "f.(X,Y)")
    elseif haskey(expr_calls, head) && nargs >= 1  # :ref/:curly/:calldecl/:(.)
        funcargslike = head == :(.) ? ex.args[2].args : ex.args[2:end]
        show_call(io, head, ex.args[1], funcargslike, indent)
    # comparison (i.e. "x < y < z")
    elseif (head == :comparison) && nargs >= 3 && (nargs & 1==1)
        comp_prec = minimum(operator_precedence, args[2:2:end])
        if comp_prec <= prec
            show_enclosed_list(io, '(', args, " ", ')', indent, comp_prec)
        else
            show_list(io, args, " ", indent, comp_prec)
        end
    # block with argument
    elseif head in (:while, :function, :if) && nargs == 2
        show_block(io, head, args[1], args[2], indent)
        print(io, "}")
    elseif head == :for
        forheader = args[1]
        forheader.head == :(=) || error("Unsupported for: $ex")
        i, range = forheader.args
        range.head == :(:) || error("Unsupported for: $ex")
        from, to = range.args
        print(io, "for(")
        show_unquoted(io, i)
        print(io, " = ")
        show_unquoted(io, from)
        print(io, "; ")
        show_unquoted(io, i)
        print(io, " <= ")
        show_unquoted(io, to)
        print(io, "; ")
        show_unquoted(io, i)
        print(io, "++)")
        show_block(io, "", [], args[2], indent)
        print(io, "}")
    elseif (head == :module) && nargs==3 && isa(args[1],Bool)
        show_block(io, args[1] ? :module : :baremodule, args[2], args[3], indent)
        print(io, "}")

    # type annotation (i.e. "::Int")
    elseif head == :(::) && nargs == 1
        print(io, ' ')
        print(io, typename(io, args[1]))

    elseif (nargs == 0 && head in (:break, :continue))
        print(io, head)

    elseif (nargs == 1 && head in (:abstract, :const)) ||
                          head in (:local,  :global, :export)
        print(io, head, ' ')
        show_list(io, args, ", ", indent)

    elseif (head === :macrocall) && nargs >= 1
        # expand macros
        show_unquoted(io, expand(ex), indent)

    elseif (head === :line) && 1 <= nargs <= 2
        show_linenumber(io, args...)

    elseif (head === :if) && nargs == 3     # if/else
        show_block(io, "if",   args[1], args[2], indent)
        show_block(io, "} else", args[3], indent)
        print(io, "}")

    elseif (head === :block) || (head === :body)
        show_block(io, "", ex, indent); print(io, "}")

    elseif (head === :new)
        constr_args = args[2:end]
        if isempty(constr_args)
            push!(constr_args, 0f0)
        end
        show_call(io, :constructor, args[1], constr_args, indent)
    elseif head === :return
        if length(args) == 1
            if args[1] != nothing
                if !(isa(args[1], Expr) && (args[1].typ == Void))
                    # if returns void, we need to omit the return statement
                    print(io, "return ")
                end
                show_unquoted(io, args[1])
            end
        else
            # ignore if empty, otherwise, LOL? What's a return with multiple args?
            if isempty(args)
                print(io, "return")
            else
                error("Unknown return Expr: $ex")
            end
        end
    elseif (head in (:meta, :inbounds, :boundscheck))
        # TODO, just ignore this? Log this? We definitely don't need it in GLSL
    elseif head == :local_memory_init
        # e.g. __local float test[100]
        T, N, name = args
        print(io, "__local ")
        show_type(io, T)
        print(io, ' ')
        show_name(io, name)
        print(io, "[$N]")
    elseif head == :local_memory
        # take the above initialized name and returns a pointer (local_memory_init & local_memory
        # are guaranteed to be emitted together)
        # and now just return &test;
        T, name = args
        print(io, "( __local ")
        show_type(io, T)
        print(io, " *)(&")
        show_name(io, name)
        print(io, ')')
    else
        println(ex)
        unsupported_expr(string(ex), line_number)
    end
    nothing
end

function show_typed_list(io, list, seperator, intent)
    for (i, elem) in enumerate(list)
        i != 1 && print(io, seperator)
        name, T = elem.args
        print(io, "    "^(intent))
        show_type(io, T)
        print(io, ' ')
        show_name(io, name)
    end
end

function show_returntype(io, method)
    T = Sugar.returntype(method)
    # void can only be used for return types in C, but in Julia it's used for all sorts of things
    # so we special case the return type and otherwise treat Void not as an inbuild.
    if T == Void
        print(io, "void")
    else
        show_type(io, T)
    end
    return
end
function Sugar.getfuncheader!(x::Union{LazyMethod{:CL}, LazyMethod{:GL}})
    if !isdefined(x, :funcheader)
        x.funcheader = if Sugar.isfunction(x)
            prototype = sprint() do io
                args = Sugar.getfuncargs(x)
                glio = CIO(io, x)
                show_returntype(glio, x)
                print(glio, ' ')
                show_unquoted(glio, x)
                print(glio, '(')
                for (i, elem) in enumerate(args)
                    if i != 1
                        print(glio, ", ")
                    end
                    name, T = elem.args
                    show_type(glio, T)
                    print(glio, ' ')
                    show_name(glio, name)
                end
                print(glio, ')')
            end
            # first is the function prototype!
            header = prototype
            prototype * ";\n" * header
        else
            ""
        end
    end
    x.funcheader
end

function Sugar.getfuncsource(x::Union{LazyMethod{:CL}, LazyMethod{:GL}})
    # TODO make this lazy as well?
    sprint() do io
        show_unquoted(CIO(io, x), Sugar.getast!(x), 0, 0)
    end
end


function Sugar.getfuncargs(x::Union{LazyMethod{:CL}, LazyMethod{:GL}})
    functype = x.signature[1]
    calltypes, slots = Sugar.to_tuple(x.signature[2]), Sugar.getslots!(x)
    n = Sugar.method_nargs(x)
    start = ifelse(Sugar.isclosure(functype), 1, 2)
    unpacked_pointers = []
    args = map(start:n) do i
        argtype, name = slots[i]
        # Slot types might be less specific, e.g. when the variable is unused it might end up as Any.
        # but generally the slot type is the correct one, especially in the context of varargs.
        if !isleaftype(argtype) && length(calltypes) <= i
            argtype = calltypes[i - 1]
        end
        if Sugar.contains_tracked_type(x, argtype)[1] && haskey(x.cache, :tracked_types)
            pointers = map(x.cache[:tracked_types][TypedSlot(i, argtype)]) do field
                ptr_typ = Sugar.get_fields_type(argtype, field[1:end-1])
                :($(last(field))::$ptr_typ)
            end
            append!(unpacked_pointers, pointers)
        end
        expr = :($(name)::$(argtype))
        expr.typ = argtype
        expr
    end
    vcat(args, unpacked_pointers)
end


include("rewriting.jl")
