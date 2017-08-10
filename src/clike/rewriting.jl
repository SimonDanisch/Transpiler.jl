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

to_index(m, T, idx::T) where T <: Integer = (idx - T(1))
function to_index(m, T, idx::Expr)
    int_type = expr_type(m, idx)
    emit_call(m, -, int_type, idx, int_type(1))
end


function rewrite_indices(m::LazyMethod, T, indices)
    map(indices) do idx
        if supports_indexing(m, T)
            return to_index(m, T, idx)
        else
            return field_symbol(m, T, idx) # indexing not supported, use getfield
        end
    end
end

function index_expression(m, expr, args, types)
    try
        isempty(types) && error("Indexing into nothing with func: $m")
        T = types[1] # type to index into
        indices, index_types = args[2:end], types[2:end]

        if types[1] <: Tuple{X} where X <: cli.Numbers
            # drop indexing into tuples of length 1. They get mapped to the element type
            # which is not very nice, but allows to map fixedsize arrays to tuples,
            # and still have fast Tuple{N, <: Numbers} types.
            return true, Sugar.rewrite_ast(m, expr.args[2])
        end
        if supports_indices(m, T, index_types)
            indices = rewrite_indices(m, T, indices)
            ret = if supports_indexing(m, T)
                typed_expr(expr.typ, :ref, args[2], indices...)
            else
                emit_call(m, getfield, expr.typ, args[2], indices...)
            end
        else
            error("Unsupported indices for type $T. Found: $index_types")
        end
    catch e
        Sugar.print_stack_trace(STDERR, m)
        rethrow(e)
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
    if f in (getfield, getindex, setindex!)
        index_expression(method, expr, expr.args[2:end], types)
    elseif f == Base.indexed_next && length(types) == 3 && isa(expr.args[3], Integer)
        # if we have a static indexed next, we unfold it into a a getindex directly
        expr.args = expr.args[1:3]
        types = types[1:2]
        replaceit, replacement = getindex_replace(li, expr, types)
        replaceit && return replacement
    elseif f == convert && length(types) == 2
        # BIG TODO, this changes semantic!!!! DONT
        if types[1] == Type{types[2]}
            return expr.args[3] # no convert needed
        else # But for now we just change a convert into a constructor call
            t = LazyMethod(li, expr.args[2])
            return similar_expr(expr, [t, expr.args[3:end]...])
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
        BF = types[1]
        # most opencl intrinsics broadcast over fixedsize arrays
        if BF <: cli.Functions && all(T-> T <: cli.Types || is_fixedsize_array(T), types[2:end])
            shift!(expr.args) # remove broadcast from call expression
            fb = Sugar.resolve_func(li, expr.args[1]) # rewrite if necessary
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
end
