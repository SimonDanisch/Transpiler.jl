# In Sugar.jl we rewrite the expressions so that they're
# closer to julias AST returned by macros.
# Here we rewrite the AST to make it easier to transpile, turning it into an
# invalid julia AST. The idea is to remove more logick from the actual printing
# to glsl string
using Sugar, DataStructures
import Sugar: similar_expr, instance, rewrite_function

is_ntuple(x) = false
is_ntuple{N, T}(x::Type{NTuple{N, T}}) = true
ntuple_length{N, T}(x::Type{NTuple{N, T}}) = N

# Functions
function rewrite_function{F}(li::CLMethod, f::F, types::ANY, expr)
    if f == broadcast
        BF = types[1]
        if BF <: cli.Functions && all(T-> T <: cli.Types, types[2:end])
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
        return expr
    elseif f == tuple
        expr.args[1] = expr.typ # use returntype
        return expr
    elseif f == convert && length(types) == 2
        # BIG TODO, this changes semantic!!!! DONT
        if types[1] == Type{types[2]}
            return expr.args[3]
        else # But for now we just change a convert into a constructor call
            return similar_expr(args, args[2:end])
        end
    elseif f == setindex! && length(types) == 3 &&
            types[1] <: CLArray && types[3] <: Integer

        idx = expr.args[4]
        idx_expr = if isa(idx, Integer)
            idx - 1
        else
            :($(idx) - 1)
        end
        ret = Expr(:(=), Expr(:ref, expr.args[2], idx_expr), expr.args[3])
        ret.typ = expr.typ
        return ret
    # homogenous tuples, translated to glsl array
    elseif f == getindex
        if length(types) == 2 && types[2] <: Integer
            T, idxT = types
            # homogenous tuples, translated to static array
            if is_ntuple(T)
                N = ntuple_length(T)
                ET = eltype(T)
                if N == 1 && ET <: cli.Numbers # Since OpenCL is really annoying we treat Tuple{T<:Number} as numbers
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
            elseif T <: CLArray
                idx = expr.args[3]
                idx_expr = if isa(idx, Integer)
                    idx - 1
                else
                    :($(expr.args[3]) - 1)
                end
                ret = Expr(:ref, expr.args[2], idx_expr)
                ret.typ = expr.typ
                return ret
            else
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
        end
    # Base.^ is in OpenCL pow
    elseif f == (^) && length(args) == 2 && all(t-> t <: cli.Numbers, types)
        expr.args[1] = cli.pow
        return expr
    # Constructors
    elseif F <: Type
        realtype = Sugar.expr_type(li, expr)
        # C/Opencl uses curly braces for constructors
        ret = Expr(:curly, realtype, expr.args[2:end]...)
        ret.typ = realtype
        return ret
    elseif F <: Function
        expr.args[1] = f
        return expr
    end
    return expr
end
