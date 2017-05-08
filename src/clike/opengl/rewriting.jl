# In Sugar.jl we rewrite the expressions so that they're
# closer to julias AST returned by macros.
# Here we rewrite the AST to make it easier to transpile, turning it into an
# invalid julia AST. The idea is to remove more logick from the actual printing
# to glsl string
import Sugar: similar_expr, instance, rewrite_function


"""

"""
function replace_slots_exctract_function(m, expr, slot2replace)
    used_functions = []
    list = Sugar.replace_expr(expr) do expr
        if isa(expr, Expr)
            if expr.head == :call
                f = Sugar.resolve_func(m, expr.args[1])
                types = Tuple{map(arg-> Sugar.expr_type(m, arg), expr.args[2:end])...}
                push!(used_functions, (f, types))
            end
        elseif isa(expr, Slot) && haskey(slot2replace, expr)
            return true, slot2replace[expr]
        end
        return false, expr
    end
    println("list : ", list)
    list[1], used_functions
end

function inline_expr(m, f, types, call_args)
    lambda = Sugar.get_lambda(code_typed, f, types)
    ast = Sugar.get_ast(lambda)
    ast = normalize_ast(ast)
    body = Expr(:block)
    append!(body.args, ast)
    nargs = length(call_args)
    tmp_assigns = []; slot2replace = Dict{SlotNumber, TypedSlot}()
    st = Sugar.slottypes(m)
    max_slot_id = length(st)

    for (i, expr) in enumerate(call_args)
        slot = TypedSlot(max_slot_id + i, expr_type(m, expr))
        slot2replace[SlotNumber(i)] = slot
        push!(tmp_assigns, :($slot = $expr))
    end
    ast, funs = replace_slots_exctract_function(m, body, slot2replace)
    unshift!(ast.args, tmp_assigns...)
    ast, funs
end
# Functions
function rewrite_function{F}(m::GLMethods, f::F, types::ANY, expr)
    if f == div && length(types) == 2 && all(x-> x <: Ints, types)
        expr.args[1] = (/)
        return expr
    elseif f == broadcast
        BF = types[1]
        if all(T-> T <: gli.Types || is_fixedsize_array(T), types[2:end])
            if BF <: gli.Functions
                shift!(expr.args) # remove broadcast from call expression
                expr.args[1] = Sugar.resolve_func(m, expr.args[1]) # rewrite if necessary
                return expr
            else
                arg_types = map(eltype, types[2:end])
                call_args = expr.args[3:end]
                @assert length(arg_types) == length(call_args)
                bf = Sugar.resolve_func(m, expr.args[2])
                ast, used_funcs = inline_expr(m, bf, arg_types, call_args)
                can_inline = all(used_funcs) do f_args
                    isintrinsic(GLMethod(f_args))
                end
                @show can_inline
                for (f, typ) in used_funcs
                    println("    ", f, " ", typ)
                end
                can_inline && return ast
            end
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
        realtype = Sugar.expr_type(m, expr)
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
