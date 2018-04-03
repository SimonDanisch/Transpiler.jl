using Transpiler
using Base.Test
import Transpiler: gli, GLMethod
using Sugar: getsource!, dependencies!
# empty caches
Transpiler.empty_caches!()

testkernel = GLMethod((test, (Float32, Float32)))
source = getsource!(testkernel)
testsource = """float test(float a, float b)
{
    float y;
    float x;
    x = sqrt(sin(a) * b) / (float)(10.0);
    y = (float)(33.0) * x + cos(b);
    return y * (float)(10.0);
}"""
@testset "test kernel" begin
    test_source(testsource, source)
end

decl = GLMethod((fortest, (Float32,)))
source = Sugar.getsource!(decl)
testsource = """float fortest(float x)
{
    float acc;
    long x1temp1;
    long i;
    acc = x;
    for(i = 1; i <= 5; i++){
        if(i == 1){
            acc = acc + x;
            continue;
        };
        if(i == 2){
            acc = acc - x;
            continue;
        };
        acc = acc + x * x;
    };
    return acc;
}"""
@testset "for loops + if elseif " begin
    test_source(testsource, source)
end

using Colors, Transpiler, StaticArrays
const Vec = SVector
const Vec3f0 = Vec{3, Float32}

struct Light{T}
    position::Vec{3,T}
    ambient::Vec{3,T}
    diffuse::Vec{3,T}
    diffuse_power::T
    specular::Vec{3,T}
    specular_power::T
end

struct Shading{T}
    ambient::Vec{3, T}
    specular::Vec{3, T}
    shininess::T
end

function blinnphong{NV, T}(V::Vec{NV, T}, N, L, color, shading, light)
    lambertian = max(dot(L, N), 0f0)
    half_direction = normalize(L .+ V)
    specular_angle = max(dot(half_direction, N), 0f0)
    specular = specular_angle ^ shading.shininess
    surf_color = (lambertian * color) .+ (specular * shading.specular)
    return light.ambient .+ surf_color
end

decl = Transpiler.GLMethod((blinnphong, (
    Vec3f0, Vec3f0, Vec3f0, Vec3f0, Shading{Float32}, Light{Float32}
)))

source = Sugar.getsource!(decl);

testsource = """vec3 blinnphong(vec3 V, vec3 N, vec3 L, vec3 color, Shading_float shading, Light_float light)
{
    vec3 surf_color;
    float specular;
    float specular_angle;
    vec3 half_direction;
    float lambertian;
    lambertian = max(dot(L, N), 0.0);
    half_direction = normalize(L + V);
    specular_angle = max(dot(half_direction, N), 0.0);
    specular = gpu_pow(specular_angle, shading.shininess);
    surf_color = lambertian * color + specular * shading.specular;
    return light.ambient + surf_color;
}"""

@testset "blinnphong" begin
    test_source(testsource, source)
end


using Transpiler, GeometryTypes
import Transpiler: emit_vertex_shader
function vert_volume(lr, test)
    out = lr
    return out, test
end

source, args = emit_vertex_shader(vert_volume, (Vec4f0, Float32))
write(STDOUT, source)
