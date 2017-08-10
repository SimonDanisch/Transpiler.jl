function Base.show(io::GLIO, x::Union{Float32, Float64})
    if isinf(x) || isnan(x)
        # TODO, can we still kinda support it?
        # There doesn't seem to be any cross platform way, though.
        error("NaN / Inf literals are not supported in GLSL")
    else
        print(io, Float64(x))
    end
end

function Sugar.gettypesource(x::GLMethods)
    T = x.signature
    tname = typename(EmptyCIO(), T)
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
                print(io, "    ", typename(EmptyCIO(), FT))
                print(io, ' ')
                print(io, c_fieldname(T, i))
                println(io, ';')
            end
        end
        println(io, "};")
        if (!isleaftype(T) || T <: Type) # emit type instances as singletons
            println(io, "const $tname TYP_INST_$tname;")
        end
    end
end

fixed_size_array_fieldname(::GLMethod, T, i) = (:x, :y, :z, :w)[i]

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

function print_gl_dependencies(io, m; filterfun = (m)-> true, funcio = io, typio = io)
    # dependencies
    deps = Sugar.dependencies!(m, true)
    deps = reverse(collect(deps))
    types = filter(Sugar.istype, deps)
    funcs = filter(Sugar.isfunction, deps)
    println(typio, "// dependant type declarations")
    for typ in types
        if !Sugar.isintrinsic(typ) && filterfun(typ)
            println(typio, Sugar.getsource!(typ))
        end
    end
    println(funcio, "// dependant function declarations")
    for func in funcs
        if !Sugar.isintrinsic(func) && filterfun(func)
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

    print_gl_dependencies(io, m, filterfun = m -> begin
        # return type will get removed if its a tuple
        RT <: Tuple || return true
        !(Sugar.istype(m) && m.signature == RT)
    end)

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

    print_gl_dependencies(io, m)

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
    take!(io.io), ret_types, Sugar.slotnames(m)[3:Sugar.method_nargs(m)]
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
    tup_geom, geom_length = if primitive_in == :points
        Tuple{geom_inT}, 1
    elseif primitive_in == :lines
        NTuple{2, geom_inT}, 2
    end
    # for type inference, geom in needs to be a tuple, while when we work with it, it's just the type itself
    m = GEOMMethod((shader, Tuple{arguments[1], tup_geom, arguments[3:end]...}))
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
        if !Sugar.isintrinsic(typ) && !(tup_geom == typ.signature)
            println(io, Sugar.getsource!(typ))
        end
    end
    # defer printing for the function and types
    io2 = GLIO(IOBuffer(), m)
    for func in funcs
        if !Sugar.isintrinsic(func) && !(Sugar.getfunction(func) == emitfunc)
            println(io2, Sugar.getsource!(func))
        end
    end
    emit_args = m.cache[:emit_func_args]
    arg_usage = "(gl_Position::Vec4f0, fragment_args::Any)"
    emit_arg1 = first(emit_args)
    usage = "you must call the emit function with: (gl_Position <: Vec4f0, fragment_args <: Any), or with the same types for every call"
    !all(x-> emit_arg1 == x, emit_args) && error(usage)
    !(first(emit_args)[1] <: Vec4f0) && error(usage)
    geometry_outT = emit_arg1[2]

    show_varying(io, geom_inT, geom_in_name, qualifier = :in, array = geom_length)
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
    # TODO better check
    if isa(x, Expr) && x.head == :return && !isempty(x.args)
        error("You're not supposed to return anything from GeometryShader. Make sure to include an empty return statement!")
    elseif isa(x, Expr) && x.head != :return
        # TODO Julia should always insert a return in the end? It might be allowed
        # to have a line number or nothing in the end... Not sure!
        error("internal error: Expr should have contained return. Found: $x")
    end
    push!(src_ast.args, Expr(:call, gli.EndPrimitive))
    emitname = snames[2]
    unshift!(src_ast.args, :($emitname::$emitfunctype))
    Base.show_unquoted(io, src_ast, 0, 0)
    src = take!(io.io)
    src, geometry_outT
end
