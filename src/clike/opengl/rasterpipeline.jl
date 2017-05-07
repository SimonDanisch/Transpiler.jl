emit_placeholder(position, fragout) = nothing

function Rasterizer(
        canvas, vertexarray, uniforms::Tuple,
        vertex_main, fragment_main;
        # geometry shader is optional, so it's supplied via kw_args
        geometry_main = nothing,
        max_primitives = 4,
        primitive_in = :points,
        primitive_out = :triangle_strip
    )
    shaders = Shader[]
    uniform_types = map(typeof, uniforms)
    argtypes = (Vertex{2, Float32}, uniform_types...)
    vsource, vertexout = emit_vertex_shader(vertex_main, argtypes)
    vshader = compile_shader(vsource, GL_VERTEX_SHADER, :particle_vert)
    push!(shaders, vshader)
    fragment_in = vertexout # we first assume vertex stage outputs to fragment stage
    if geometry_main != nothing
        argtypes = (typeof(emit_placeholder), vertexout, uniform_types...)
        gsource, geomout = emit_geometry_shader(
            geometry_main, argtypes,
            max_primitives = max_primitives,
            primitive_in = primitive_in,
            primitive_out = primitive_out
        )
        gshader = compile_shader(gsource, GL_GEOMETRY_SHADER, :particle_geom)
        push!(shaders, gshader)
        fragment_in = geomout # rewire if geometry shader is present
    end

    argtypes = (fragment_in, uniform_types...)
    fsource, fragout = emit_fragment_shader(fragment_main, argtypes)
    fshader = compile_shader(fsource, GL_FRAGMENT_SHADER, :particle_frag)
    push!(shaders, fshader)

    program = compile_program(shaders...)

    uniform_idx = glGetUniformBlockIndex(program, "_gensymed_uniforms")
    glUniformBlockBinding(program, uniform_idx, 0)
    glBindBufferBase(GL_UNIFORM_BUFFER, 0, uniform_buff.buffer.id)

end
