function rewrite_backend_specific{F}(m::GLMethods, f::F, types, expr)
    if f == (erf) && length(types) == 1 && all(t-> t <: AbstractFloat, types)
        expr.args[1] = LazyMethod(m, gl_erf, types)
        return true, expr
    elseif f == (erfc) && length(types) == 1 && all(t-> t <: AbstractFloat, types)
        expr.args[1] = LazyMethod(m, gl_erfc, types)
        return true, expr
    # StaticArrays.norm is in OpenGL length
    elseif f == norm && length(types) == 1 && all(t-> is_fixedsize_array(t), types)
        expr.args[1] = LazyMethod(m, length, types)
        return true, expr
    elseif f in (muladd, fma) && length(types) == 3 && all(t-> t <: cli.Floats, types)
        expr.args[1] = LazyMethod(m, gl_fma, types)
        return true, expr
    # Constructors
    else
        false, expr
    end
end
