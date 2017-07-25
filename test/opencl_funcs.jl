using Transpiler
using Base.Test
import Transpiler: cli, CLMethod
using Sugar: getsource!, dependencies!

function mapkernel(f, a, b, c)
    gid = cli.get_global_id(0) + Cuint(1)
    c[gid] = f(a[gid], b[gid])
    return
end

# empty caches
Transpiler.empty_caches!()

args = (typeof(+), cli.CLArray{Float32, 1}, cli.CLArray{Float32, 1}, cli.CLArray{Float32, 1})
cl_mapkernel = CLMethod((mapkernel, args))
source = getsource!(cl_mapkernel)
mapsource = """void mapkernel_1(Base123 f, __global float * restrict  a, __global float * restrict  b, __global float * restrict  c)
{
    uint gid;
    gid = (get_global_id)(0) + (uint){1};
    float _ssavalue_0;
    _ssavalue_0 = (a)[gid - 0x00000001] + (b)[gid - 0x00000001];
    (c)[gid - 0x00000001] = _ssavalue_0;
    ;
}"""
@testset "map kernel" begin
    @test source == mapsource
    deps = dependencies!(cl_mapkernel, true)
    @test length(deps) == 8
    deps_test = [
        UInt32,
        (+, Tuple{UInt32, UInt32}),
        (cli.get_global_id, Tuple{Int64}),
        (+, Tuple{Float32, Float32}),
        Float32,
        (-, Tuple{UInt32, UInt32}),
        typeof(+),
        cli.CLArray{Float32,1}
    ]
    for elem in deps
        @test elem.signature in deps_test
    end
end

#Broadcast
Base.@propagate_inbounds broadcast_index(arg, shape, i) = arg
Base.@propagate_inbounds function broadcast_index{T, N}(
        arg::AbstractArray{T, N}, shape::NTuple{N, Integer}, i
    )
    @inbounds return arg[i]
end

# The implementation of prod in base doesn't play very well with current
# transpiler. TODO figure out what Core._apply maps to!
_prod{T}(x::NTuple{1, T}) = x[1]
_prod{T}(x::NTuple{2, T}) = x[1] * x[2]

function broadcast_kernel(A, f, sz, arg1, arg2)
    i = cli.get_global_id(0) + Cuint(1)
    @inbounds if i <= _prod(sz)
        A[i] = f(
            broadcast_index(arg1, sz, i),
            broadcast_index(arg2, sz, i),
        )
    end
    return
end

args = (cli.CLArray{Float32, 1}, typeof(+), Tuple{UInt32}, cli.CLArray{Float32, 1}, Float32)
cl_mapkernel = CLMethod((broadcast_kernel, args))
source = getsource!(cl_mapkernel)
broadcastsource = """void broadcast_kernel_5(__global float * restrict  A, Base123 f, uint sz, __global float * restrict  arg1, float arg2)
{
    uint i;
    i = (get_global_id)(0) + (uint){1};
    ;
    if(i <= (_prod_2)(sz)){
        float _ssavalue_0;
        _ssavalue_0 = (broadcast_index_3)(arg1, sz, i) + (broadcast_index_4)(arg2, sz, i);
        (A)[i - 0x00000001] = _ssavalue_0;
    };
    ;
    ;
}"""
@testset "broadcast kernel" begin
    @test source == broadcastsource
    deps = dependencies!(cl_mapkernel, true)
    @test length(deps) == 13
    deps_test = [
        UInt32,
        (+,Tuple{UInt32,UInt32}),
        (cli.get_global_id, Tuple{Int64}),
        (<=,Tuple{UInt32,UInt32}),
        (_prod,Tuple{Tuple{UInt32}}),
        (+,Tuple{Float32,Float32}),
        (broadcast_index,Tuple{cli.CLArray{Float32,1},Tuple{UInt32},UInt32}),
        (broadcast_index,Tuple{Float32,Tuple{UInt32},UInt32}),
        (-,Tuple{UInt32,UInt32}),
        cli.CLArray{Float32,1},
        typeof(+),
        Tuple{UInt32},
        Float32
    ]
    for elem in deps
        @test elem.signature in deps_test
    end
end

decl = CLMethod((fortest, (Float32,)))
source = Sugar.getsource!(decl)
#TODO remove xxtempx4, which is unused now...
target_source = """float fortest_6(float x)
{
    float acc;
    int xtempx_4;
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
@test target_source == source
