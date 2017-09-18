function rewrite_backend_specific{F}(m::CLMethod, f::F, types, expr)
    li = m.parent
    if f == (ifelse) && length(types) == 3 && types[1] == Bool && all(x-> x <: cli.Types || is_fixedsize_array(x), types[2:3])
        _types = (types[2], types[3], types[1])
        sm = LazyMethod(li, cli.cl_select, _types)
        expr.args = [sm, expr.args[3], expr.args[4], expr.args[2]]
        return true, expr
    elseif f == abs && length(types) == 1 && all(t-> t <: cli.Floats, types)
        expr.args[1] = LazyMethod(li, fabs, types)
        return true, expr
    elseif f == muladd && length(types) == 3 && all(t-> t <: cli.Floats, types)
        expr.args[1] = LazyMethod(li, fma, types)
        return true, expr
    elseif Symbol(f) == :synchronize_threads
        expr.args[1] = LazyMethod(li, cli.barrier, (Cuint,))
        expr.args[2] = :CLK_LOCAL_MEM_FENCE
        return true, expr
    else
        return false, expr
    end
end
