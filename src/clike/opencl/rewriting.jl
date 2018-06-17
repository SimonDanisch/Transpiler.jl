extract_val(::Type{Val{X}}) where X = X
function rewrite_backend_specific(m::CLMethod, f::F, types, expr) where F
    li = m.parent
    if f == (ifelse) && length(types) == 3 && types[1] == Bool && all(x-> x <: cli.Types || is_fixedsize_array(x), types[2:3])
        _types = (types[2], types[3], types[1])
        sm = LazyMethod(li, cli.cl_select, _types)
        expr.args = [sm, expr.args[3], expr.args[4], expr.args[2]]
        return true, expr
    elseif f == abs && length(types) == 1 && all(t-> t <: cli.Floats, types)
        expr.args[1] = LazyMethod(li, fabs, types)
        return true, expr
    elseif f == mod && length(types) == 2 && all(t-> t <: cli.Ints, types)
        expr.args[1] = LazyMethod(li, %, types)
        return true, expr
    elseif f == rem && length(types) == 2 && all(t-> t <: cli.Ints, types)
        expr.args[1] = LazyMethod(li, %, types)
        return true, expr
    elseif f == muladd && length(types) == 3 && all(t-> t <: cli.Floats, types)

        expr.args[1] = LazyMethod(li, fma, types)
        return true, expr
    elseif Symbol(f) == :synchronize_threads
        expr.args[1] = LazyMethod(li, cli.barrier, (Cuint,))
        expr.args[2] = :CLK_LOCAL_MEM_FENCE
        return true, expr
    elseif Symbol(f) == Symbol("GPUArrays.LocalMemory")
        if length(expr.args) != 5
            error("LocalMemory needs to be used as: LocalMemory(state, T, N), with T & N being constants / static function parameters")
        end
        T = Sugar.rewrite_ast.(m, expr.args[3])
        N = expr.args[4].args[1]
        if !(isa(T, Type)) #&& (isa(N, Val{<:Integer}) || isa(N, Val{NTuple{N, Integer}} where N)))
             error("LocalMemory needs to be used as: LocalMemory(T, N), with T being a type and Val{N} where N <: Union{Integer, Tuple}. Found: $T $N")
        end
        name = gensym("localmem")
        ret = Expr(:local_memory, T, name)
        ret.typ = cli.LocalPointer{T}
        inline = Sugar.InlineNode([Expr(:local_memory_init, T, prod(extract_val(N)), name)], ret)
        return true, inline
    else
        return false, expr
    end
end
