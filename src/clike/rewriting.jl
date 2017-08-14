using Sugar: expr_type, typed_expr, getfunction, similar_expr
"""
Emit a LazyMethod call Expr
"""
function emit_call(m, fun, typ, args...)
    typs = map(x-> expr_type(m, x), args)
    m2 = LazyMethod(m, fun, typs)
    typed_expr(typ, :call, m2, args...)
end


"""
Types that are accessible only via getfield,
e.g. Tuples that are translated to structs - so in Julia
They'll be addressable via getindex, but in another language it must be getfield.
"""
supports_indexing(m::LazyMethod, T) = false

"""
Types that are only indexable via static indices.
"""
needs_static_index(m::LazyMethod, T) = true

supports_indices(m::LazyMethod, T, index_types) = false

to_index(m, t, idx::T) where T <: Integer = (idx - T(1))
function to_index(m, T, idx)
    int_type = expr_type(m, idx)
    emit_call(m, -, int_type, idx, int_type(1))
end


function rewrite_indices(m::LazyMethod, T, indices)
    map(indices) do idx
        if supports_indexing(m, T)
            return to_index(m, T, idx)
        else
            return c_fieldname(m, T, idx) # indexing not supported, use getfield
        end
    end
end

function index_expression(m, expr, args, types)
    mparent = m.parent
    is_setindex = getfunction(m) == setindex!
    isempty(types) && error("Indexing into nothing with func: $m")
    T = types[1] # type to index into
    isimmutable(T) && is_setindex && error("Can't use setindex! on immutable. Found: $T with $types")
    indices, val, index_types = if is_setindex
        args[3:end], args[2], types[3:end]
    else
        args[2:end], nothing, types[2:end]
    end

    if T <: Tuple{X} where X <: cli.Numbers
        # drop indexing into tuples of length 1. They get mapped to the element type
        # which is not very nice, but allows to map fixedsize arrays to tuples,
        # and still have fast Tuple{N, <: Numbers} types.
        return Sugar.rewrite_ast(mparent, expr.args[2])
    end
    if supports_indices(mparent, T, index_types)
        indices = rewrite_indices(mparent, T, indices)
        ret = if supports_indexing(mparent, T)
            indexing = typed_expr(expr.typ, :ref, args[1], indices...)
            if is_setindex
                indexing = :($indexing = $val)
            end
            indexing
        else
            if (T <: Tuple || is_fixedsize_array(T)) # Julia inbuilds allowing getindex, but need to use getfield in C
                emit_call(mparent, getfield, expr.typ, args[1], indices...)
            else
                expr.args[1] = m # leave getindex
                expr
            end
        end
        return ret
    else
        expr.args[1] = m # leave getindex
        return expr
    end
end

function emit_constructor(m::LazyMethod, realtype, args)
    Sugar.typed_expr(realtype, :curly, m, args...)
end

function Sugar.rewrite_function(method::Union{LazyMethod{:CL}, LazyMethod{:GL}}, expr)
    f, _types = method.signature
    types = Sugar.to_tuple(_types)
    li = method.parent
    F = typeof(f)
    # first give the backends a chance to do backend specific rewriting
    rewritten, expr = rewrite_backend_specific(method, f, types, expr)
    rewritten && return expr
    # if nothing got rewritten, do the backend independant rewrites:
    if f in (getindex, setindex!)
        return index_expression(method, expr, expr.args[2:end], types)
    elseif f == getfield && length(types) == 2 && types[2] <: Integer
        return emit_call(
            method, getfield,
            Sugar.expr_type(method, expr),
            expr.args[1], c_fieldname(method, types[1], expr.args[2])
        )
    elseif f == Base.indexed_next && length(types) == 3 && isa(expr.args[3], Integer)
        # if we have a static indexed next, we unfold it into a a getindex directly
        expr.args = expr.args[1:3]
        types = types[1:2]
        replaceit, replacement = getindex_replace(li, expr, types)
        replaceit && return replacement
    elseif f == convert && length(types) == 2
        # BIG TODO, this changes semantic!!!! DONT
        if types[1] == Type{types[2]}
            return Sugar.rewrite_ast(li, expr.args[3]) # no convert needed
        else # But for now we just change a convert into a constructor call
            realtype = Sugar.unspecialized_type(Sugar.expr_type(li, expr.args[2]))
            target_type = Sugar.unspecialized_type(Sugar.expr_type(li, expr.args[3]))
            if is_native_type(li, realtype) && is_native_type(li, target_type)
                constr_m = LazyMethod(li, realtype)
                return emit_constructor(constr_m, realtype, expr.args[3:end])
            else
                return emit_call(li, convert, realtype, realtype, expr.args[3])
            end
        end
    # Constructors
    elseif F <: Type || f == tuple
        realtype = Sugar.expr_type(li, expr)
        args = expr.args[2:end]
        if isempty(args) && sizeof(realtype) == 0
            push!(args, 0f0) # there are no empty types, so if empty, insert default
        end
        # C/Opencl uses curly braces for constructors
        constr_m = LazyMethod(li, realtype)
        return emit_constructor(constr_m, realtype, args)
    elseif f == broadcast
        fb = Sugar.resolve_func(li, expr.args[2]) # rewrite if necessary
        # most shared intrinsics broadcast over fixedsize arrays
        if (fb in functions) && all(t-> is_native_type(li, t), types[2:end])
            shift!(expr.args) # remove broadcast from call expression
            expr.args[1] = LazyMethod(li, fb, types[2:end])
            return expr
        end
        return expr
    # div is / in c like languages
    elseif f == div && length(types) == 2 && all(x-> x <: cli.Ints, types)
        expr.args[1] = LazyMethod(li, (/), types)
        return expr
    # Base.^ is pow in C
    elseif f == (^) && length(types) == 2 && all(t-> t <: cli.Numbers, types)
        expr.args[1] = LazyMethod(li, pow, types)
        return expr
    elseif F <: Function
        expr.args[1] = method
        return expr
    end
    return expr
end
