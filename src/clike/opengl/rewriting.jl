# In Sugar.jl we rewrite the expressions so that they're
# closer to julias AST returned by macros.
# Here we rewrite the AST to make it easier to transpile, turning it into an
# invalid julia AST. The idea is to remove more logick from the actual printing
# to glsl string
import Sugar: similar_expr, instance, rewrite_function

# Functions
function rewrite_function{F}(li::GLMethods, f::F, types::ANY, expr)
    if f == div && length(types) == 2 && all(x-> x <: Ints, types)
        expr.args[1] = (/)
        return expr
    elseif f == broadcast
        BF = types[1]
        if BF <: gli.Functions && all(T-> T <: gli.Types || is_fixedsize_array(T), types[2:end])
            shift!(expr.args) # remove broadcast from call expression
            expr.args[1] = Sugar.resolve_func(li, expr.args[1]) # rewrite if necessary
            return expr
        end
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
            return similar_expr(expr, expr.args[2:end])
        end
    elseif f == setindex! && length(types) == 3 &&
            types[1] <: GLDeviceArray && all(x-> x <: Integer, types[3:end])

        idxs = expr.args[4:end]
        idx_expr = map(idxs) do idx
            if isa(idx, Integer)
                idx - 1
            else
                :($(idx) - 1)
            end
        end
        ret = Expr(:(=), Expr(:ref, expr.args[2], idx_expr...), expr.args[3])
        ret.typ = expr.typ
        return ret
    elseif f == getfield
        idx = expr.args[3]
        if isa(idx, Integer)
            name = Symbol(c_fieldname(types[1], idx))
            ret = Expr(:call, getfield, expr.args[2], name)
            ret.typ = expr.typ
            return ret
        elseif isa(idx, Symbol)
            expr.args[1] = getfield
            return expr
        else
            error("Indexing like: $expr not implemented")
        end
    elseif f == getindex
        if !isempty(types)
            T = types[1]
            indexchars = (:x, :y, :z, :w)
            # Indexing with e.g. Vec(1,2)
            if (length(types) == 2 &&
                    is_fixedsize_array(types[2]) && (eltype(types[2]) <: Integer) &&
                    (is_ntuple(T) || is_fixedsize_array(T)))
                idx = expr.args[3].args[2:end]

                idxsym = Symbol(join(map(idx) do idx
                    if !isa(idx, Integer)
                        error("Only static indices are allowed. Found: $idx")
                    end
                    indexchars[idx]
                end, ""))
                ret = Expr(:call, getfield, expr.args[2], idxsym)
                ret.typ = expr.typ
                return ret
            # homogenous tuples, translated to static array
            elseif (length(types) == 2 && types[2] <: Integer) &&
                    (is_ntuple(T) || is_fixedsize_array(T))

                N = fixed_array_length(T)
                ET = eltype(T)
                if N == 1 && ET <: Numbers # Since OpenCL is really annoying we treat Tuple{T<:Number} as numbers
                    return expr.args[2]
                end
                # todo replace static indices
                idx = expr.args[3]
                if is_fixedsize_array(T)
                    if !isa(idx, Integer)
                        error("Only static indices are allowed for small vectors!")
                    end
                    idxsym = indexchars[idx]
                    ret = Expr(:call, getfield, expr.args[2], idxsym)
                    ret.typ = expr.typ
                    return ret
                end
                idx_expr = if isa(idx, Integer)
                    idx - 1
                else
                    :($(expr.args[3]) - 1)
                end

                ret = Expr(:ref, expr.args[2], idx_expr)
                ret.typ = expr.typ
                return ret
            elseif T <: AbstractArray
                # lets hope it has a sensible getindex
                expr.args[1] = getindex
                return expr
            else
                idx = expr.args[3]
                idx_expr = if isa(idx, Integer)
                    name = Symbol(c_fieldname(T, idx))
                    ret = Expr(:call, getfield, expr.args[2], name)
                    ret.typ = expr.typ
                    return ret
                else
                    # Will need some dynamic field lookup!
                    error(
                        "Only static getindex into composed types allowed for now!
                        Found: $T with $(types[2:end])"
                    )
                end
                error("indexing into $T not implemented")
                ret.typ = expr.typ
                return ret
            end
        end
    # Base.^ is in OpenGL pow
    elseif f == (^) && length(types) == 2 && all(t-> t <: Numbers, types)
        expr.args[1] = pow
        return expr
    # StaticArrays.norm is in OpenGL length
    elseif f == norm && length(types) == 1 && all(t-> is_fixedsize_array(t), types)
        expr.args[1] = length
        return expr
    # Constructors
    elseif F <: Type
        realtype = Sugar.expr_type(li, expr)
        # C/Opencl uses curly braces for constructors
        ret = Expr(:call, realtype, expr.args[2:end]...)
        ret.typ = realtype
        return ret
    elseif F <: Function
        expr.args[1] = f
        return expr
    end
    return expr
end
