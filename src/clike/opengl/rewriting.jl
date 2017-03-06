# In Sugar.jl we rewrite the expressions so that they're
# closer to julias AST returned by macros.
# Here we rewrite the AST to make it easier to transpile, turning it into an
# invalid julia AST. The idea is to remove more logick from the actual printing
# to glsl string

using Sugar, DataStructures
import Sugar: similar_expr, instance, rewrite_function

const GLMethod = LazyMethod{:GL}

# Functions
function rewrite_function{F <: Function}(li::GLMethod, f::F, types::ANY, expr)
    expr.args[1] = f
    expr
end

# Constructors
function rewrite_function{T}(li::GLMethod, f::Type{T}, types::ANY, expr)
    realtype = Sugar.expr_type(li, expr)
    expr.args[1] = realtype
    expr
end

# constructors... #TODO refine input types
function rewrite_function{T <: gli.Types}(li::GLMethod, f::Type{T}, types::ANY, expr)
    expr.args[1] = T
    expr
end

# homogenous tuples, translated to glsl array
function rewrite_function{N, T, I <: Integer}(
        li::GLMethod, f::typeof(getindex), types::Type{Tuple{NTuple{N, T}, I}}, expr
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

function rewrite_function{T <: gli.Vecs, I <: gli.int}(
        li::GLMethod, f::typeof(getindex), types::Type{Tuple{T, I}}, expr
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

function rewrite_function{V1 <: gli.Vecs, V2 <: gli.Vecs}(
        li::GLMethod, f::Type{V1}, types::Type{Tuple{V2}}, expr
    )
    expr.args[1] = f # substitute constructor
    expr
end

function rewrite_function(li::GLMethod, f::typeof(tuple), types::ANY, expr)
    expr.args[1] = expr.typ # use returntype
    expr
end

function rewrite_function(li::GLMethod, f::typeof(broadcast), types::ANY, expr)
    tuptypes = (types.parameters...)
    F = tuptypes[1]
    if F <: gli.Functions && all(T-> T <: gli.Types, tuptypes[2:end])
        shift!(expr.args) # remove broadcast from call expression
        expr.args[1] = Sugar.resolve_func(li, expr.args[1]) # rewrite if necessary
        return expr
    end
    # if all(x-> x <: gli.Types, args)
    #     ast = Sugar.sugared(f, types, code_lowered)
    #     funcs = extract_funcs(ast)
    #     types = extract_types(ast)
    #     if all(is_intrinsic, funcs) && all(x-> (x <: gli.Types), types)
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
