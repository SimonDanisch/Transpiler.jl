# In Sugar.jl we rewrite the expressions so that they're
# closer to julias AST returned by macros.
# Here we rewrite the AST to make it easier to transpile, turning it into an
# invalid julia AST. The idea is to remove more logick from the actual printing
# to glsl string
using Sugar, DataStructures, StaticArrays
import Sugar: rewrite_function
using Sugar: typed_expr, similar_expr, instance

function emit_call(m, fun, typ, args...)
    typs = map(x-> expr_type(m, x), args)
    m2 = LazyMethod(m, fun, typs)
    typed_expr(typ, :call, m2, args...)
end

function getindex_replace(m, expr, _types)
    types = Sugar.to_tuple(_types)
    if !isempty(types)
        T = types[1]
        # homogenous tuples, translated to static array
        if (length(types) == 2 && types[2] <: Integer) && cli.is_fixedsize_array(T)
            N = cli.fixed_array_length(T)
            ET = eltype(T)
            if N == 1 && ET <: cli.Numbers # Since OpenCL is really annoying we treat Tuple{T<:Number} as numbers
                return false, expr
            end
            # todo replace static indices
            idx = expr.args[3]
            if cli.is_fixedsize_array(T)
                if !isa(idx, Integer)
                    error("Only static indices are allowed for small vectors!. Found: $expr")
                end
                idxsym = Symbol("s$(idx - 1)")
                ret = emit_call(m, getfield, expr.typ, expr.args[2], idxsym)
                return true, ret
            end
            idx_expr = if isa(idx, Integer)
                idx - 1
            else
                int_type = expr_type(m, idx)
                emit_call(m, -, int_type, expr.args[3], int_type(1))
            end
            ret = typed_expr(expr.typ, :ref, expr.args[2], idx_expr)
            return true, ret
        elseif T <: CLDeviceArray && all(x-> x <: Integer, types[2:end])
            idxs = expr.args[3:end]
            idx_expr = map(idxs) do idx
                if isa(idx, Integer)
                    idx - 1
                else
                    int_type = expr_type(m, idx)
                    emit_call(m, -, int_type, expr.args[3], int_type(1))
                end
            end
            ret = typed_expr(expr.typ, :ref, expr.args[2], idx_expr)
            return true, ret
        elseif types[1] <: Tuple{X} where X <: cli.Numbers
            return true, Sugar.rewrite_ast(m, expr.args[2]) # don't index, tuples of length 1 get mapped to the primitive itself -.-
        else
            idx = expr.args[3]
            idx_expr = if isa(idx, Integer)
                # TODO have better fieldname method
                name = Symbol(c_fieldname(T, idx))
            else
                # Will need some dynamic field lookup!
                Sugar.print_stack_trace(STDERR, m)
                error(
                    "Only static getindex into composed types allowed for now!
                    Found: $T with $(types[2:end])"
                )
            end
            ret = emit_call(m, getfield, expr.typ, expr.args[2], idx_expr)
            return true, ret
        end
    end
    return false, expr
end
# Functions


function rewrite_function(method::CLMethod, expr)
    Sugar.istype(method) && return expr
    f, _types = method.signature
    types = Sugar.to_tuple(_types)
    li = method.parent
    F = typeof(f)
    if f == div && length(types) == 2 && all(x-> x <: cli.Ints, types)
        expr.args[1] = LazyMethod(li, (/), types)
        return expr
    elseif f == broadcast
        BF = types[1]
        # most opencl intrinsics broadcast over fixedsize arrays
        if BF <: cli.Functions && all(T-> T <: cli.Types || is_fixedsize_array(T), types[2:end])
            shift!(expr.args) # remove broadcast from call expression
            fb = Sugar.resolve_func(li, expr.args[1]) # rewrite if necessary
            expr.args[1] = LazyMethod(li, fb, types[2:end])
            return expr
        end
        return expr
    elseif f == convert && length(types) == 2
        # BIG TODO, this changes semantic!!!! DONT
        if types[1] == Type{types[2]}
            return expr.args[3] # no convert needed
        else # But for now we just change a convert into a constructor call
            t = LazyMethod(li, expr.args[2])
            return similar_expr(expr, [t, expr.args[3:end]...])
        end
    elseif f == setindex! && length(types) >= 3 &&
            types[1] <: CLDeviceArray && all(x-> x <: Integer, types[3:end])
        idxs = expr.args[4:end]
        idx_expr = map(idxs) do idx
            if isa(idx, Integer)
                idx - 1
            else
                int_type = expr_type(li, idx)
                emit_call(li, -, int_type, idx, int_type(1))
            end
        end
        ret = Expr(:(=), Expr(:ref, expr.args[2], idx_expr...), expr.args[3])
        ret.typ = expr.typ
        return ret
    elseif f == getfield && length(types) == 2 && isa(expr.args[3], Integer)
        expr.args[1] = LazyMethod(li, getfield, types)
        idx = expr.args[3]
        expr.args[3] = if is_fixedsize_array(types[1])
            Symbol("s$(idx - 1)")
        else
            Symbol(c_fieldname(types[1], idx))
        end
        return expr
    elseif f == Base.indexed_next && length(types) == 3 && isa(expr.args[3], Integer)
        # if we have a static indexed next, we unfold it into a a getindex directly
        expr.args = expr.args[1:3]
        types = types[1:2]
        replaceit, replacement = getindex_replace(li, expr, types)
        replaceit && return replacement
    elseif f == getindex
        replaceit, replacement = getindex_replace(li, expr, types)
        replaceit && return replacement
    # Base.^ is in OpenCL pow
    elseif f == (^) && length(types) == 2 && all(t-> t <: cli.Numbers, types)
        expr.args[1] = LazyMethod(li, pow, types)
        return expr
    elseif f == (ifelse) && length(types) == 3 && types[1] == Bool && all(x-> x <: cli.Types || is_fixedsize_array(x), types[2:3])
        _types = (types[2], types[3], types[1])
        sm = LazyMethod(li, cli.select, _types)
        expr.args = [sm, expr.args[3], expr.args[4], expr.args[2]]
        return expr
    elseif f == abs && length(types) == 1 && all(t-> t <: cli.Floats, types)
        expr.args[1] = LazyMethod(li, fabs, types)
        return expr
    elseif f == muladd && length(types) == 3 && all(t-> t <: cli.Floats, types)
        expr.args[1] = LazyMethod(li, fma, types)
        return expr
    # Constructors
    elseif F <: Type || f == tuple
        realtype = Sugar.expr_type(li, expr)
        args = expr.args[2:end]
        if isempty(args) && sizeof(realtype) == 0
            push!(args, 0f0) # there are no empty types, so if empty, insert default
        end
        # C/Opencl uses curly braces for constructors
        ret = typed_expr(realtype, :curly, LazyMethod(li, realtype), args...)
        return ret
    elseif F <: Function
        expr.args[1] = method
        return expr
    end
    return expr
end
