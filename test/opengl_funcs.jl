using Transpiler
using Transpiler: GLSLTranspiler
import Transpiler.GLSLTranspiler.gli
import Transpiler.GLSLTranspiler.CLFunction
import Transpiler.GLSLTranspiler.GlobalInvocationID
using Sugar, FileIO
using ModernGL, Reactive, GLFW, StaticArrays
import Sugar: ssavalue_name, ASTIO, LazyMethod
import Sugar: getsource!, dependencies!, istype, isfunction, getfuncargs

const Vec = SVector
const Vec3f0 = SVector{3, Float32}

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

program = CLFunction(broadcast_kernel, (x, test, b, c))
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


using Colors

immutable Light{T}
    position::Vec{3,T}
    ambient::Vec{3,T}
    diffuse::Vec{3,T}
    diffuse_power::T
    specular::Vec{3,T}
    specular_power::T
end

immutable Shading{T}
    ambient::Vec{3, T}
    specular::Vec{3, T}
    shininess::T
end

function blinnphong{NV, T}(light, L::Vec{NV, T}, N, V, color, shading)
    diff_coeff = max(dot(L,N), T(0.0f0))
    # specular coefficient
    H = normalize(L + V)
    spec_coeff = max(dot(H,N), T(0.0f0)) ^ (shading.shininess)
    if diff_coeff <= T(0.0)
        spec_coeff = T(0.0)
    elseif diff_coeff <= T(0.2)
        # some nonesense to test elseif
        spec_coeff *= T(2.0)
        spec_coeff += T(1.0)
    else
        spec_coeff = spec_coeff
        return L
    end
    # final lighting model
    return Vec3f0(
        light.ambient .* shading.ambient +
        light.diffuse .* light.diffuse_power .* color * diff_coeff +
        light.specular .* light.specular_power .* shading.specular * spec_coeff
    )
end

decl = Transpiler.GLSLTranspiler.GLMethod((blinnphong, (
    Light{Float32}, Vec3f0, Vec3f0, Vec3f0, Vec3f0, Shading{Float32}
)))

println(getsource!(decl))



function mysum1(A)
    s = eltype(A)(0)
    for a in A
        s += a
    end
    return s
end
function julia{T, T2}(z::T, maxiter::T2)
    c = T(-0.5, 0.75)
    for n = 1:maxiter
        if abs2(z) > 4.0
            return T2(n - 1)
        end
        z = z * z + c
    end
    return maxiter
end

method = Transpiler.GLSLTranspiler.GLMethod((julia, (Complex64, Int)))
for dep in dependencies!(method)
    try
    println(getsource!(dep))
    end
end
