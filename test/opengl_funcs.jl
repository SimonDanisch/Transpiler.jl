using Transpiler
using Base.Test
import Transpiler: gli, GLMethod
using Sugar: getsource!, dependencies!
# empty caches
Transpiler.empty_caches!()

cl_mapkernel = GLMethod((test, (Float32, Float32)))
source = getsource!(cl_mapkernel)
testsource = """float test(float a, float b)
{
    float y;
    float x;
    x = sqrt(sin(a) * b) / float(10.0);
    y = float(33.0) * x + cos(b);
    return y * float(10.0);
}"""
@testset "test kernel" begin
    @test source == testsource
end

decl = GLMethod((fortest, (Float32,)))
source = Sugar.getsource!(decl)
#TODO remove xxtempx4, which is unused now...

testsource = """float fortest(float x)
{
    float acc;
    int xxtempx4;
    int i;
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
    @test testsource == source
end

using Colors, Transpiler, StaticArrays
const Vec = SVector
const Vec3f0 = Vec{3, Float32}

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

decl = Transpiler.GLMethod((blinnphong, (
    Light{Float32}, Vec3f0, Vec3f0, Vec3f0, Vec3f0, Shading{Float32}
)))
deps = Sugar.dependencies!(decl, true);
for dep in deps
    if !Sugar.isintrinsic(dep)
        println(Sugar.getsource!(dep))
    end
end
SVector{3,Float32}
# ast = Sugar.getast!(decl)
# isa(ast.args[12].args[2].args[1], Type)
# println(ast)
source = Sugar.getsource!(decl)
println(source)

testsource = """
vec3 blinnphong(Light_float light, vec3 L, vec3 N, vec3 V, vec3 color, Shading_float shading)
{
    float spec_coeff;
    vec3 H;
    float diff_coeff;
    diff_coeff = max(dot(L, N), float(0.0));
    H = normalize(L + V);
    spec_coeff = pow(max(dot(H, N), float(0.0)), shading.shininess);
    if(diff_coeff <= float(0.0)){
        spec_coeff = float(0.0);
    } else{
        if(diff_coeff <= float(0.2)){
            spec_coeff = spec_coeff * float(2.0);
            spec_coeff = spec_coeff + float(1.0);
        };
        spec_coeff = spec_coeff;
        return L;
    };
    return vec3(_45(light.ambient, shading.ambient) + _45(_45(light.diffuse, light.diffuse_power), color) * diff_coeff + _45(_45(light.specular, light.specular_power), shading.specular) * spec_coeff);
}"""
@testset "blinnphong" begin
    @test source == testsource
end

function test(a, b, c, s)
    x = a .* b
    y = (c .* dot(b, c)) .* (b * s) +
    c .* a .* b * s
    x +
end
function test(a, b, c, s)
    a .* b +
    c .* dot(b, c) .* b * s +
    c .* a .* b * s
end
using BenchmarkTools
bench = @benchmark test(
    $(SVector(1f0, 2f0, 3f0)),
    $(SVector(1f0, 1f0, 1f0)),
    $(SVector(1f0, 1f0, 1f0)),
    $(7.42f0)
)
