#=
This file contains functions taken and modified from Julia/base/show.jl
license of show.jl:
Copyright (c) 2009-2016: Jeff Bezanson, Stefan Karpinski, Viral B. Shah, and other contributors:
https://github.com/JuliaLang/julia/contributors
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
=#


import Base: indent_width, uni_ops, expr_infix_wide
import Base: all_ops, expr_calls, show_block
import Base: show_list, show_enclosed_list, operator_precedence
import Base: is_quoted, is_expr, expr_infix_any
import Base: show_call, show_unquoted, show



show_linenumber(io::GLIO, line)       = print(io, " // line ", line,':')
show_linenumber(io::GLIO, line, file) = print(io, " // ", file, ", line ", line, ':')


# don't print f0 TODO this is a Float32 hack
function show(io::GLIO, x::Float32)
    print(io, Float64(x))
end
function show_unquoted(io::GLIO, sym::Symbol, ::Int, ::Int)
    print(io, Symbol(symbol_hygiene(io, sym)))
end

function show_unquoted(io::GLIO, ex::GlobalRef, ::Int, ::Int)
    # TODO Why is Base.x suddenly == GPUArrays.GLBackend.x
    #if ex.mod == GLSLIntrinsics# || ex.mod == GPUArrays.GLBackend
        print(io, ex.name)
    #else
    #    error("No non Intrinsic GlobalRef's for now!: $ex")
    #end
end

# show a normal (non-operator) function call, e.g. f(x,y) or A[z]
function show_call(io::GLIO, head, func, func_args, indent)
    op, cl = expr_calls[head]
    if head == :ref
        show_unquoted(io, func, indent)
    else
        show_function(io, func, map(x->expr_type(io.method, x), func_args))
    end
    if head == :(.)
        print(io, '.')
    end
    if !isempty(func_args) && isa(func_args[1], Expr) && func_args[1].head == :parameters
        print(io, op)
        show_list(io, func_args[2:end], ", ", indent)
        print(io, "; ")
        show_list(io, func_args[1].args, ", ", indent)
        print(io, cl)
    else
        show_enclosed_list(io, op, func_args, ", ", cl, indent)
    end
end


function Base.show_unquoted(io::GLIO, ssa::SSAValue, ::Int, ::Int)
    print(io, Sugar.ssavalue_name(ssa))
end

# show a block, e g if/for/etc
function show_block(io::GLIO, head, args::Vector, body, indent::Int)
    if isempty(args)
        print(io, head, '{')
    else
        print(io, head, '(')
        show_list(io, args, ", ", indent)
        print(io, "){")
    end

    ind = head === :module || head === :baremodule ? indent : indent + indent_width
    exs = (is_expr(body, :block) || is_expr(body, :body)) ? body.args : Any[body]
    for (i, ex) in enumerate(exs)
        sep = i == 1 ? "" : ";"
        print(io, sep, '\n', " "^ind)
        show_unquoted(io, ex, ind, -1)
    end
    print(io, ";\n", " "^indent)
end


function show_unquoted(io::GLIO, ex::Expr, indent::Int, prec::Int)
    line_number = 0 # TODO include line numbers
    head, args, nargs = ex.head, ex.args, length(ex.args)
    # dot (i.e. "x.y"), but not compact broadcast exps
    if head == :(.) && !is_expr(args[2], :tuple)
        show_unquoted(io, args[1], indent + indent_width)
        print(io, '.')
        if is_quoted(args[2])
            show_unquoted(io, unquoted(args[2]), indent + indent_width)
        else
            print(io, '(')
            show_unquoted(io, args[2], indent + indent_width)
            print(io, ')')
        end

    # variable declaration TODO might this occure in normal code?
    elseif head == :(::) && nargs == 2
        show_type(io, args[2])
        print(io, ' ')
        show_name(io, args[1])
    # infix (i.e. "x<:y" or "x = y")

    elseif (head in expr_infix_any && (nargs == 2) || (head == :(:)) && nargs == 3)
        func_prec = operator_precedence(head)
        head_ = head in expr_infix_wide ? " $head " : head
        if func_prec <= prec
            show_enclosed_list(io, '(', args, head_, ')', indent, func_prec, true)
        else
            show_list(io, args, head_, indent, func_prec, true)
        end

    # function call
    elseif head === :call && nargs >= 1
        f = first(args)
        func_args = args[2:end]
        func_arg_types = map(x->expr_type(io.method, x), func_args)
        fname = Symbol(functionname(io, f, func_arg_types))
        if fname == :getfield && nargs == 3
            show_unquoted(io, args[2], indent) # type to be accessed
            print(io, '.')
            show_unquoted(io, args[3], indent)
        else
            # TODO handle getfield
            func_prec = operator_precedence(fname)
            # scalar multiplication (i.e. "100x")
            if (
                    fname === :* &&
                    length(func_args) == 2 && isa(func_args[1], Real) &&
                    isa(func_args[2], Symbol)
                )
                if func_prec <= prec
                    show_enclosed_list(io, '(', func_args, "", ')', indent, func_prec)
                else
                    show_list(io, func_args, "", indent, func_prec)
                end

            # unary operator (i.e. "!z")
            elseif isa(fname, Symbol) && fname in uni_ops && length(func_args) == 1
                print(io, fname)
                if isa(func_args[1], Expr) || func_args[1] in all_ops
                    show_enclosed_list(io, '(', func_args, ",", ')', indent, func_prec)
                else
                    show_unquoted(io, func_args[1])
                end

            # binary operator (i.e. "x + y")
            elseif func_prec > 0 # is a binary operator
                na = length(func_args)
                if (na == 2 || (na > 2 && fname in (:+, :++, :*))) && all(!isa(a, Expr) || a.head !== :... for a in func_args)
                    sep = " $f "
                    if func_prec <= prec
                        show_enclosed_list(io, '(', func_args, sep, ')', indent, func_prec, true)
                    else
                        show_list(io, func_args, sep, indent, func_prec, true)
                    end
                elseif na == 1
                    # 1-argument call to normally-binary operator
                    op, cl = expr_calls[head]
                    show_unquoted(io, f, indent)
                    show_enclosed_list(io, op, func_args, ",", cl, indent)
                else
                    show_call(io, head, f, func_args, indent)
                end

            # normal function (i.e. "f(x,y)")
            else
                show_call(io, head, f, func_args, indent)
            end
        end
    # other call-like expressions ("A[1,2]", "T{X,Y}", "f.(X,Y)")
    elseif haskey(expr_calls, head) && nargs >= 1  # :ref/:curly/:calldecl/:(.)
        funcargslike = head == :(.) ? ex.args[2].args : ex.args[2:end]
        show_call(io, head, ex.args[1], funcargslike, indent)
    # comparison (i.e. "x < y < z")
    elseif (head == :comparison) && nargs >= 3 && (nargs & 1==1)
        comp_prec = minimum(operator_precedence, args[2:2:end])
        if comp_prec <= prec
            show_enclosed_list(io, '(', args, " ", ')', indent, comp_prec)
        else
            show_list(io, args, " ", indent, comp_prec)
        end

    # block with argument
    elseif head in (:while, :function, :if) && nargs == 2
        show_block(io, head, args[1], args[2], indent)
        print(io, "}")
    elseif head == :for
        forheader = args[1]
        forheader.head == :(=) || error("Unsupported for: $ex")
        i, range = forheader.args
        range.head == :(:) || error("Unsupported for: $ex")
        from, to = range.args
        print(io, "for(")
        show_unquoted(io, i)
        print(io, " = ")
        show_unquoted(io, from)
        print(io, "; ")
        show_unquoted(io, i)
        print(io, " <= ")
        show_unquoted(io, to)
        print(io, "; ")
        show_unquoted(io, i)
        print(io, "++)")
        show_block(io, "", [], args[2], indent)
        print(io, "}")
    elseif (head == :module) && nargs==3 && isa(args[1],Bool)
        show_block(io, args[1] ? :module : :baremodule, args[2], args[3], indent)
        print(io, "}")

    # empty return (i.e. "function f() return end")
    elseif head == :return && nargs == 1 && (args[1] === nothing)
        # ignore empty return

    # type annotation (i.e. "::Int")
    elseif head == :(::) && nargs == 1
        print(io, ' ')
        print(io, typename(io, args[1]))

    elseif (nargs == 0 && head in (:break, :continue))
        print(io, head)

    elseif (head === :macrocall) && nargs >= 1
        # Use the functional syntax unless specifically designated with prec=-1
        show_unquoted(io, expand(ex), indent)

    elseif (head === :typealias) && nargs == 2
        print(io, "typedef ")
        show_list(io, args, ' ', indent)

    elseif (head === :line) && 1 <= nargs <= 2
        show_linenumber(io, args...)

    elseif (head === :if) && nargs == 3     # if/else
        show_block(io, "if",   args[1], args[2], indent)
        show_block(io, "} else", args[3], indent)
        print(io, "}")

    elseif (head === :block) || (head === :body)
        show_block(io, "", ex, indent); print(io, "}")

    elseif ((head === :&)#= || (head === :$)=#) && length(args) == 1
        print(io, head)
        a1 = args[1]
        parens = (isa(a1, Expr) && a1.head !== :tuple) || (isa(a1,Symbol) && isoperator(a1))
        parens && print(io, "(")
        show_unquoted(io, a1)
        parens && print(io, ")")



    elseif (head === :return)
        if length(args) == 1
            # return Void must not return anything in GLSL
            if !((isa(args[1], Expr) && args[1].typ == Void) || args[1] == nothing)
                print(io, "return ")
            end
            show_unquoted(io, args[1])
        elseif isempty(args)
            # ignore return if no args or void
        else
            error("What dis return? $ex")
        end
    elseif head == :inbounds # ignore
    elseif (head === :meta)
        # TODO, just ignore this? Log this? We definitely don't need it in GLSL
    else
        println(ex)
        unsupported_expr(string(ex), line_number)
    end
    nothing
end


function Sugar.getfuncheader!(x::GLMethod)
    if !isdefined(x, :funcheader)
        x.funcheader = if Sugar.isfunction(x)
            sprint() do io
                args = Sugar.getfuncargs(x)
                glio = GLIO(io, x)
                show_type(glio, Sugar.returntype(x))
                print(glio, ' ')
                show_function(glio, x.signature...)
                print(glio, '(')
                for (i, elem) in enumerate(args)
                    if i != 1
                        print(glio, ", ")
                    end
                    name, T = elem.args
                    show_type(glio, T)
                    print(glio, ' ')
                    show_name(glio, name)
                end
                print(glio, ')')
            end
        else
            ""
        end
    end
    x.funcheader
end

function Sugar.getfuncsource(x::GLMethod)
    # TODO make this lazy as well?
    sprint() do io
        show_unquoted(GLIO(io, x), Sugar.getast!(x), 0, 0)
    end
end

function Sugar.gettypesource(x::GLMethod)
    T = x.signature
    tname = typename(EmptyGLIO(), T)
    sprint() do io
        print(io, "struct $tname{\n")
        nf = nfields(T)
        fields = []
        if nf == 0 # structs can't be empty
            # we use bool as a short placeholder type.
            # TODO, are there cases where bool is no good?
            println(io, "float empty; // structs can't be empty")
        else
            for i in 1:nf
                FT = fieldtype(T, i)
                print(io, "    ", typename(EmptyGLIO(), FT))
                print(io, ' ')
                print(io, c_fieldname(T, i))
                println(io, ';')
            end
        end
        println(io, "};")
    end
end

function functionname(io::GLIO, f, types)
    if isa(f, Type) || isa(f, Expr)
        # This should only happen, if the function is actually a type
        if isa(f, Expr)
            f = f.typ
        end
        return _typename(io, f)
    end
    method = try
        LazyMethod(io.method, f, types)
    catch e
        error("Couldn't create function $f with $types")
    end
    f_sym = Symbol(typeof(f).name.mt.name)
    if Sugar.isintrinsic(method)
        return f_sym # intrinsic operators don't need hygiene!
    end
    str = if supports_overloading(io)
        string(f_sym)
    else
        string(f_sym, '_', signature_hash(types))
    end
    symbol_hygiene(io, str)
end