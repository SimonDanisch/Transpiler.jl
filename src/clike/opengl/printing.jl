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
using GeometryTypes


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
    # TODO disregarding modules doesn't seem to be a good idea.
    # Thought about just appending the module to the name, but this doesn't work
    # very well, since Julia allows itself quite a bit of freedom, when it's attaching
    # the module to a name or not. E.g. depending on where things where created, you might get
    # GPUArrays.GPUArray, or Visualize.GPUArrays.GPUArray.
    print(io, ex.name)
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
            #TODO when we ignore a line, show_block will still print a semicolon
            # ignore return if no args or void
        else
            error("What dis return? $ex")
        end
    elseif (head === :meta) || head == :inbounds
        # TODO, just ignore this? Log this? We definitely don't need it in GLSL
    else
        println(ex)
        error(string(ex, " ", line_number))
    end
    nothing
end

function show_typed_list(io, list, seperator, intent)
    for (i, elem) in enumerate(list)
        i != 1 && print(io, seperator)
        name, T = elem.args
        print(io, "    "^(intent))
        show_type(io, T)
        print(io, ' ')
        show_name(io, name)
    end
end
function Sugar.getfuncheader!(x::GLMethods)
    if !isdefined(x, :funcheader)
        x.funcheader = if Sugar.isfunction(x)
            sprint() do io
                args = Sugar.getfuncargs(x)
                glio = GLIO(io, x)
                show_type(glio, Sugar.returntype(x))
                print(glio, ' ')
                show_function(glio, x.signature...)
                print(glio, '(')
                show_typed_list(glio, args, ", ", 0)
                print(glio, ')')
            end
        else
            ""
        end
    end
    x.funcheader
end

function Sugar.getfuncsource(x::GLMethods)
    # TODO make this lazy as well?
    sprint() do io
        show_unquoted(GLIO(io, x), Sugar.getast!(x), 0, 0)
    end
end

function Sugar.gettypesource(x::GLMethods)
    T = x.signature
    tname = typename(EmptyGLIO(), T)
    sprint() do io
        println(io, "// Julia name: $T")
        print(io, "struct $tname{\n")
        nf = nfields(T)
        fields = []
        if nf == 0 # structs can't be empty
            # we use bool as a short placeholder type.
            # TODO, are there cases where bool is no good?
            println(io, "    float empty; // structs can't be empty")
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
        Base.showerror(STDERR, e)
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



function glsl_gensym(name)
    # TODO track all symbols and actually emit a unique symbol
    Symbol(string("_gensymed_", name))
end

function show_vertex_input(io, T, name, start_idx = 0)
    args = typed_type_fields(T)
    idx = start_idx
    for arg in args
        fname, T = arg.args
        print(io, "layout (location = $idx) in ")
        show_type(io, T)
        print(io, ' ')
        show_name(io, name)
        print(io, '_')
        show_name(io, fname)
        println(io, ';')
        idx += 1
    end
end


array_string(array) = array == -1 ? "" : array == 0 ? "[]" : "[$array]"
"""
array can be -1 for now array, 0 for an unfixed length and
everything above will be treated as a fixed length
"""
function show_varying(io, T, name; qualifier::Symbol = :in, array::Int = -1)
    array_expr = array_string(array)
    print(io, "$qualifier ")
    show_type(io, T)
    print(io, ' ')
    show_name(io, name)
    println(io, array_expr, ';')
end

function show_interface_block(
        io, T, name;
        qualifier::Symbol = :uniform, layout = ["std140"],
        array::Int = -1, blockname = glsl_gensym(name)
    )
    array_expr = array_string(array)
    if !isempty(layout)
        print(io, "layout ")
        print(io, '(', join(layout, ", "), ") ")
    end
    print(io, "$qualifier ", blockname, "{\n    ")
    show_type(io, T)
    print(io, ' ')
    show_name(io, name)
    println(io, array_expr,";\n};")
end

function print_dependencies(io, m; funcio = io, typio = io)
    # dependencies
    deps = Sugar.dependencies!(m, true)
    println("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
    println("_--------------------------------------")
    println("_--------------------------------------")
    println("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
    deps = reverse(collect(deps))
    types = filter(Sugar.istype, deps)
    funcs = filter(Sugar.isfunction, deps)
    println(typio, "// dependant type declarations")
    for typ in types
        if !Sugar.isintrinsic(typ)
            println(typio, Sugar.getsource!(typ))
        end
    end
    println(funcio, "// dependant function declarations")
    for func in funcs
        if !Sugar.isintrinsic(func)
            println(funcio, Sugar.getsource!(func))
        end
    end
end

function show_uniforms(io, arg_names, arg_types)
    println(io, "// uniform inputs:")
    for (i, (name, T)) in enumerate(zip(arg_names, arg_types))
        if T <: gli.GLTexture
            print(io, "uniform ")
            show_type(io, T)
            print(io, ' ')
            show_name(io, name)
            print(io, ';')
        else
            show_interface_block(
                io, T, name, qualifier = :uniform,
                blockname = glsl_gensym("UniformArg$i")
            )
        end
        println(io)
    end
end

function emit_vertex_shader(shader::Function, arguments::Tuple)
    m = Transpiler.GLMethod((shader, Tuple{arguments...}))
    io = Transpiler.GLIO(IOBuffer(), m)
    nargs = Sugar.method_nargs(m)
    snames = Sugar.slotnames(m)
    stypes = Sugar.slottypes(m)
    RT = Sugar.returntype(m)

    vertex_type = first(arguments)
    vertex_name = snames[2]

    arg_types = arguments[2:end]
    arg_names = snames[3:nargs]

    # get body ast
    Sugar.getcodeinfo!(m) # make sure codeinfo is present
    ast = Sugar.sugared(m.signature..., code_typed)
    for i in (nargs + 1):length(stypes)
        T = stypes[i]
        slot = SlotNumber(i)
        push!(m.decls, slot)
        name = snames[i]
        tmp = :($name::$T)
        tmp.typ = T
        unshift!(ast.args, tmp)
    end

    ret_expr = pop!(ast.args)
    ast.typ = Void

    # TODO glsl version string?!
    println(io, "#version 330")

    print_dependencies(io, m)

    # Vertex in block
    println(io, "// vertex input:")
    Transpiler.show_vertex_input(io, vertex_type, vertex_name)
    vertex_args = map(Transpiler.typed_type_fields(vertex_type)) do arg
        fname, T = arg.args
        Symbol(string(vertex_name, '_', fname))
    end
    vertex_expr = Expr(:call, vertex_type, vertex_args...)
    vertex_expr.typ = vertex_type
    unshift!(ast.args, :($vertex_name = $vertex_expr))
    unshift!(ast.args, :($vertex_name::$vertex_type))


    # uniform block
    show_uniforms(io, arg_names, arg_types)

    # output
    usage = """
    You need to return a tuple from vertex shader of the form:
    `(gl_Position::Vec4f0, vertex_out::Any)`
    If you don't want to write to gl_Position, it's okay to just return vertex_out::Any
    """
    vertex_out_T, vertex_out_expr = if (RT <: Tuple)
        ret_types = Sugar.to_tuple(RT)
        !(first(ret_types) <: Vec4f0) || length(ret_types) != 2 && error(usage)
        # ret_expr: :(return (...,))
        out_tuple = ret_expr.args[1].args[2:end]
        length(out_tuple) != 2 && error(usage)
        glposition, vertexout = out_tuple
        # write to gl_Position
        push!(ast.args, :(gl_Position = $(glposition)))
        ret_types[2], vertexout
    else
        RT, ret_expr.args[1]
    end

    vertexsym = :vertex_out
    # println(io, Sugar.getsource!(GLMethod(vertex_out_T)))
    show_varying(io, vertex_out_T, vertexsym, qualifier = :out)
    # write to out
    push!(ast.args, :($vertexsym = $vertex_out_expr))

    println(io)
    println(io, "// vertex main function:")
    println(io, "void main()")
    # we already declared these, so hint to transpiler not to declare them again
    push!(m.decls, vertexsym, :gl_Position, vertex_name)
    src_ast = Sugar.rewrite_ast(m, ast)
    Base.show_unquoted(io, src_ast, 0, 0)
    take!(io.io), vertex_out_T
end



function emit_fragment_shader(shader, arguments)
    m = Transpiler.GLMethod((shader, Tuple{arguments...}))
    io = Transpiler.GLIO(IOBuffer(), m)
    nargs = Sugar.method_nargs(m)
    snames = Sugar.slotnames(m)
    stypes = Sugar.slottypes(m)
    RT = Sugar.returntype(m)

    fragmentinT = first(arguments)
    fragment_name = snames[2] # 1 is self

    arg_types = arguments[2:end]
    arg_names = snames[3:nargs]
    # get body ast
    Sugar.getcodeinfo!(m) # make sure codeinfo is present
    ast = Sugar.sugared(m.signature..., code_typed)
    for i in (nargs + 1):length(stypes)
        T = stypes[i]
        slot = SlotNumber(i)
        push!(m.decls, slot)
        name = snames[i]
        tmp = :($name::$T)
        tmp.typ = T
        unshift!(ast.args, tmp)
    end

    ret_expr = pop!(ast.args)
    ast.typ = Void

    println(io, "#version 330")

    print_dependencies(io, m)

    # uniform block
    show_uniforms(io, arg_names, arg_types)

    show_varying(io, fragmentinT, fragment_name, qualifier = :in)

    usage = """
    You need to return a tuple from fragment shader: (Color0::Vec4f0, Color1::Union{Numbers, StaticVector}, ...),
    which will write into the respective framebuffer.
    For one framebuffer, you can also just return the color!
    """
    ret_types = if !(RT <: Tuple)
        if RT <: Vec4f0
            (RT,)
        else
            error(usage)
        end
    else
        Sugar.to_tuple(RT)
    end
    # wrong return types
    !all(x-> x <: StaticVector || x <: gli.Numbers, ret_types) && error(usage)
    out_tuple = ret_expr.args[1].args[2:end]
    for (i, elem) in enumerate(out_tuple)
        framebuffname = glsl_gensym("color$(i - 1)")
        print(io, "layout (location = $(i - 1)) out ")
        show_type(io, Sugar.expr_type(m, elem))
        print(io, ' ')
        show_name(io, framebuffname)
        println(io, ';')
        push!(ast.args, :($framebuffname = $elem))
        push!(m.decls, framebuffname)
    end
    println(io)
    println(io, "// fragment main function:")
    println(io, "void main()")
    src_ast = Sugar.rewrite_ast(m, ast)
    Base.show_unquoted(io, src_ast, 0, 0)
    take!(io.io), ret_types
end


function Sugar.getast!(m::GEOMMethod)
    if !isdefined(m, :ast)
        emitfunc = m.cache[:emitfunc]; emit_func_args = m.cache[:emit_func_args]
        outname = m.cache[:outname]
        Sugar.getcodeinfo!(m) # make sure codeinfo is present
        nargs = Sugar.method_nargs(m)
        expr = Sugar.sugared(m.signature..., code_typed)
        st = Sugar.slottypes(m)
        for (i, T) in enumerate(st)
            slot = SlotNumber(i)
            push!(m.decls, slot)
            if i > nargs # if not defined in arguments, define in body
                name = Sugar.slotname(m, slot)
                tmp = :($name::$T)
                tmp.typ = T
                unshift!(expr.args, tmp)
            end
        end
        expr.typ = Sugar.returntype(m)
        expr = Sugar.rewrite_ast(m, expr)
        emit_call = Expr(:call, gli.EmitVertex)
        emit_call.typ = Void
        expr = first(Sugar.replace_expr(expr) do expr
            if isa(expr, Expr) && expr.head == :call
                func = expr.args[1]
                if func == emitfunc
                    args = (map(x-> Sugar.expr_type(m, x), expr.args[2:end])...)
                    push!(emit_func_args, args)
                    push!(m.decls, :gl_Position)
                    push!(m.decls, outname)
                    pos_expr = :(gl_Position = $(expr.args[2]))
                    fragout_expr = :($outname = $(expr.args[3]))
                    return true, (pos_expr, fragout_expr, emit_call)
                end
            end
            false, expr
        end)
        m.ast = expr
    end
    m.ast
end

function Sugar.isintrinsic(x::GEOMMethod)
    if Sugar.isfunction(x)
        f = Sugar.getfunction(x)
        Sugar.isintrinsic(f) ||
            gli.glintrinsic(x.signature...) ||
            f == x.cache[:emitfunc]
    else
        gli.glintrinsic(x.signature)
    end
end


function emit_geometry_shader(
        shader, arguments::Tuple;
        max_primitives = 4,
        primitive_in = :points,
        primitive_out = :triangle_strip
    )
    emitfunctype = first(arguments)
    emitfunc = Sugar.instance(emitfunctype)
    geom_inT = arguments[2]
    # for type inference, geom in needs to be a tuple, while when we work with it, it's just the type itself
    m = GEOMMethod((shader, Tuple{arguments[1], Tuple{geom_inT}, arguments[3:end]...}))
    geom_out_name = :geom_out

    m.cache[:emitfunc] = emitfunc; m.cache[:emit_func_args] = []
    m.cache[:outname] = geom_out_name

    io = GLIO(IOBuffer(), m)

    nargs = Sugar.method_nargs(m)
    snames = Sugar.slotnames(m)
    stypes = Sugar.slottypes(m)

    geom_in_name = snames[3]

    arg_types = arguments[3:end]
    arg_names = snames[4:nargs]

    # get body ast
    ast = Sugar.getast!(m)

    println(io, "#version 330")
    println(io, "layout($primitive_in) in;")
    println(io, "layout($primitive_out, max_vertices = $max_primitives) out;")

    # dependencies
    deps = reverse(collect(Sugar.dependencies!(m, true)))
    types = filter(Sugar.istype, deps)
    funcs = filter(Sugar.isfunction, deps)
    println(io, "// dependant type declarations")
    for typ in types
        if !Sugar.isintrinsic(typ) && !(Tuple{geom_inT} == typ.signature)
            println(io, Sugar.getsource!(typ))
        end
    end
    # defer printing for the function and types
    io2 = GLIO(IOBuffer(), m)
    for func in funcs
        if !Sugar.isintrinsic(func) && !(Sugar.getfunction(func) == emitfunc)
            println(func.signature)
            println(io2, Sugar.getsource!(func))
        end
    end
    emit_args = m.cache[:emit_func_args]
    arg_usage = "(gl_Position::Vec4f0, fragment_args::Any)"
    emit_arg1 = first(emit_args)
    usage = "you must call the emit function with: (gl_Position <: Vec4f0, fragment_args <: Any), with the same types for every call"
    !all(x-> emit_arg1 == x, emit_args) && error(usage)
    !(first(emit_args)[1] <: Vec4f0) && error(usage)
    geometry_outT = emit_arg1[2]

    show_varying(io, geom_inT, geom_in_name, qualifier = :in, array = 0)
    show_varying(io, geometry_outT, geom_out_name, qualifier = :out)

    # uniform block
    show_uniforms(io, arg_names, arg_types)


    println(io, "// dependant function declarations")
    write(io.io, take!(io2.io))
    println(io)
    println(io, "// geometry main function:")
    println(io, "void main()")
    src_ast = Sugar.rewrite_ast(m, ast)
    x = pop!(src_ast.args)
    if !isa(x, Expr) && x.head != :return
        # TODO Julia should always insert a return in the end? It might be allowed
        # to have a line number or nothing in the end... Not sure!
        error("internal error: Expr should have contained return. Found: $x")
    end
    push!(src_ast.args, Expr(:call, gli.EndPrimitive))
    emitname = snames[2]
    unshift!(src_ast.args, :($emitname::$emitfunctype))
    Base.show_unquoted(io, src_ast, 0, 0)
    take!(io.io), geometry_outT
end
