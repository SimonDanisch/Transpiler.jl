# In Sugar.jl we rewrite the expressions so that they're
# closer to julias AST returned by macros.
# Here we rewrite the AST to make it easier to transpile, turning it into an
# invalid julia AST. The idea is to remove more logick from the actual printing
# to glsl string

using Sugar, DataStructures
import Sugar: similar_expr, instance, rewrite_function

# Functions
function rewrite_function{F <: Function}(li::CLMethod, f::F, types::ANY, expr)
    expr.args[1] = f
    expr
end

# Constructors
function rewrite_function{T}(li::CLMethod, f::Type{T}, types::ANY, expr)
    realtype = Sugar.expr_type(li, expr)
    # C/Opencl uses curly braces for constructors  
    ret = Expr(:curly, realtype, expr.args[2:end]...)
    ret.typ = realtype
    ret
end

# constructors... #TODO refine input types
function rewrite_function{T <: cli.Types}(li::CLMethod, f::Type{T}, types::ANY, expr)
    expr.args[1] = T
    expr
end
# constructors... #TODO refine input types
function rewrite_function{T <: cli.Numbers}(li::CLMethod, f::typeof(^), types::Tuple{T, T}, expr)
    expr.args[1] = cli.pow
    expr
end
# homogenous tuples, translated to glsl array
is_pointer_type(T) = true
function rewrite_function{T, I <: Integer}(
        li::CLMethod, f::typeof(getindex), types::Type{Tuple{T, I}}, expr
    )
    # todo replace static indices
    idx = expr.args[3]
    idx_expr = if isa(idx, Integer)
        name = Symbol(c_fieldname(T, idx))
    else
        # Will need some dynamic field lookup!
        error("Only static getindex into composed types allowed for now!")
    end
    if is_pointer_type(T)
        ret = Expr(:(->), expr.args[2], idx_expr)
    else
        error("Lul, index into $T?")
    end
    ret.typ = expr.typ
    return ret
end

# homogenous tuples, translated to glsl array
function rewrite_function{N, T, I <: Integer}(
        li::CLMethod, f::typeof(getindex), types::Type{Tuple{NTuple{N, T}, I}}, expr
    )
    if N == 1 && T <: cli.Numbers # Since OpenCL is really annoying we treat Tuple{T<:Number} as numbers
        return expr.args[2]
    end
    # todo replace static indices
    idx = expr.args[3]
    idx_expr = if isa(idx, Integer)
        idx - 1
    else
        :($(expr.args[3]) - 1)
    end
    ret = Expr(:ref, expr.args[2], idx_expr)
    ret.typ = expr.typ
    return ret
end
function rewrite_function{N, T, I <: Integer}(
        li::CLMethod, f::typeof(getindex), types::Type{Tuple{CLArray{N, T}, I}}, expr
    )
    # todo replace static indices
    idx = expr.args[3]
    idx_expr = if isa(idx, Integer)
        idx - 1
    else
        :($(expr.args[3]) - 1)
    end
    ret = Expr(:ref, expr.args[2], idx_expr)
    ret.typ = expr.typ
    return ret
end
function rewrite_function{T <: CLArray, Val, I <: Integer}(
        li::CLMethod, f::typeof(setindex!), types::Type{Tuple{T, Val, I}}, expr
    )
    # todo replace static indices
    idx = expr.args[4]
    idx_expr = if isa(idx, Integer)
        idx - 1
    else
        :($(idx) - 1)
    end
    ret = Expr(:(=), Expr(:ref, expr.args[2], idx_expr), expr.args[3])
    ret.typ = expr.typ
    return ret
end
function rewrite_function{T <: cli.Vecs, I <: cli.int}(
        li::CLMethod, f::typeof(getindex), types::Type{Tuple{T, I}}, expr
    )
    # todo replace static indices
    idx = expr.args[3]
    if isa(idx, Integer) # if value is inlined
        field = (:x, :y, :z, :w)[idx]
        return similar_expr(expr, [getfield, expr.args[2], field])
    else
        ret = Expr(:ref, expr.args[2:end]...)
        ret.typ = expr.typ
        return ret
    end
end

function rewrite_function{V1 <: cli.Vecs, V2 <: cli.Vecs}(
        li::CLMethod, f::Type{V1}, types::Type{Tuple{V2}}, expr
    )
    expr.args[1] = f # substitute constructor
    expr
end

function rewrite_function{T}(
        li::CLMethod, f::typeof(convert), types::Type{Tuple{Type{T}, T}}, expr
    )
    # drop convert of same type
    expr.args[3]
end
# rewrite intrinsic converts into
function rewrite_function{T1 <: cli.Types, T2 <: cli.Types}(
        li::CLMethod, f::typeof(convert), types::Type{Tuple{Type{T1}, T2}}, expr
    )
    # drop convert
    similar_expr(args, args[2:end])
end

function rewrite_function(li::CLMethod, f::typeof(tuple), types::ANY, expr)
    expr.args[1] = expr.typ # use returntype
    expr
end

function rewrite_function(li::CLMethod, f::typeof(broadcast), types::ANY, expr)
    tuptypes = (types.parameters...)
    F = tuptypes[1]
    if F <: cli.Functions && all(T-> T <: cli.Types, tuptypes[2:end])
        shift!(expr.args) # remove broadcast from call expression
        expr.args[1] = Sugar.resolve_func(li, expr.args[1]) # rewrite if necessary
        return expr
    end
    # if all(x-> x <: cli.Types, args)
    #     ast = Sugar.sugared(f, types, code_lowered)
    #     funcs = extract_funcs(ast)
    #     types = extract_types(ast)
    #     if all(is_intrinsic, funcs) && all(x-> (x <: cli.Types), types)
    #         # if we only have intrinsic functions, numbers and vecs we can rewrite
    #         # this broadcast to be direct function calls. since the intrinsics will
    #         # already be broadcasted
    #         ast = glsl_ast(broadcast, types)
    #         false, :broadcast
    #     end
    # end
    expr.args[1] = f
    expr
end
