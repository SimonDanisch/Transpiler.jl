using Transpiler
using Transpiler: GLSLTranspiler
import Transpiler.GLSLTranspiler.gli
import Transpiler.GLSLTranspiler.ComputeProgram
import Transpiler.GLSLTranspiler.GlobalInvocationID
using Sugar, GeometryTypes, FileIO
using GLWindow, GLAbstraction, ModernGL, Reactive, GLFW, GeometryTypes
import Sugar: ssavalue_name, ASTIO, get_slottypename, get_type, LazyMethod
import Sugar: getsource!, dependencies!, istype, isfunction, getfuncargs


function test{T}(a::T, b)
    x = sqrt(sin(a) * b) / T(10.0)
    y = T(33.0)x + cos(b)
    y * T(10.0)
end
function broadcast_index{T}(arg::gli.GLArray{T, 2}, shape::NTuple{2, Any}, idx)
    arg[idx]
end
broadcast_index(arg, shape, idx) = arg

function broadcast_kernel{T}(A::gli.GLArray{T, 2}, f, B, C)
    idx = NTuple{2, Int}(GlobalInvocationID())
    sz  = size(A)
    A[idx] = f(broadcast_index(B, sz, idx), broadcast_index(C, sz, idx))
    return
end
using GLAbstraction
ctx = GLWindow.create_glcontext(major = 4, minor = 3, visible = false)
x = Texture(Float32, (1024, 3));
b = Texture(Float32, (1024, 3));
c = Texture(Float32, (1024, 3));
args = (x, test, b, c);

gltypes = Transpiler.GLSLTranspiler.to_glsl_types(args)
decl = Transpiler.GLSLTranspiler.GLMethod((broadcast_kernel, gltypes))

#
# MacroTools.prewalk(Sugar.getast!(decl)) do expr
#     if isa(expr, Expr) && expr.head != :block
#         println(expr)
#         println(Sugar.expr_type(decl, expr))
#     end
#     expr
# end

for elem in Sugar.dependencies!(decl, true)
    #if !Sugar.isintrinsic(elem)
        println(elem.signature)
    #end
end
deps = Sugar.dependencies!(decl, true)

program = ComputeProgram(broadcast_kernel, (x, test, b, c))
args = (b, (1, 1), (1, 1));
gltypes = Transpiler.GLSLTranspiler.to_glsl_types(args)
decl = Transpiler.GLSLTranspiler.GLMethod((broadcast_index, gltypes))


function write2framebuffer(color, id)
    fragment_color = color
    if (color.a > 0.5)
        gl_FragDepth = gl_FragCoord()[3]
    else
        gl_FragDepth = 1.0
    end
    @fragmentout begin
        fragment_color = color
        fragment_groupid = id
    end
end

function fragmentshader(image, uv, swizzle, id_idx)
    write2framebuffer(
        image[uv[swizzle]],
        objectid
    )
end

function vertexshader(pvm, objectid, vertex)
    @vertexout begin
        id_idx = uvec2(objectid, gl_VertexID())
        position_camspace = pvm * Point4f0(vertex[Position])
        uv = vertex[UV]
    end
end
