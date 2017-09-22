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

args = (typeof(+), cli.GlobalPointer{Float32}, cli.GlobalPointer{Float32}, cli.GlobalPointer{Float32})
cl_mapkernel = CLMethod((mapkernel, args))
source = Sugar.getsource!(cl_mapkernel)
mapsource = """void mapkernel_1(Base123 f, __global float *  a, __global float *  b, __global float *  c)
{
    uint gid;
    gid = get_global_id(0) + (uint)(1);
    float _ssavalue_0;
    _ssavalue_0 = (a)[gid - 0x00000001] + (b)[gid - 0x00000001];
    (c)[gid - 0x00000001] = _ssavalue_0;
    ;
}"""

@testset "map kernel" begin
    @test source == mapsource
    deps = dependencies!(cl_mapkernel, true)
    deps_test = [
        Int64,
        UInt32,
        (+, Tuple{UInt32, UInt32}),
        (cli.get_global_id, Tuple{Int64}),
        (+, Tuple{Float32, Float32}),
        Float32,
        (-, Tuple{UInt32, UInt32}),
        typeof(+),
        cli.GlobalPointer{Float32},
        (UInt32, Tuple{Int64}),
        typeof(mapkernel),
    ]
    @test length(deps) == length(deps_test)
    for elem in deps
        @test elem.signature in deps_test
    end
end

#Broadcast
Base.@propagate_inbounds broadcast_index(arg, shape, i) = arg
Base.@propagate_inbounds function broadcast_index{T, N}(
        arg::cli.GlobalPointer{T}, shape::NTuple{N, Integer}, i
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

args = (cli.GlobalPointer{Float32}, typeof(+), Tuple{UInt32}, cli.GlobalPointer{Float32}, Float32)
cl_mapkernel = CLMethod((broadcast_kernel, args))
source = getsource!(cl_mapkernel)
broadcastsource = """void broadcast_kernel_5(__global float *  A, Base123 f, uint sz, __global float *  arg1, float arg2)
{
    uint i;
    i = get_global_id(0) + (uint)(1);
    ;
    if(i <= _prod_2(sz)){
        float _ssavalue_0;
        _ssavalue_0 = broadcast_index_3(arg1, sz, i) + broadcast_index_4(arg2, sz, i);
        (A)[i - 0x00000001] = _ssavalue_0;
    };
    ;
    ;
}"""
@testset "broadcast kernel" begin
    @test source == broadcastsource
    deps = dependencies!(cl_mapkernel, true)
    deps_test = [
        typeof(broadcast_kernel),
        UInt32,
        Int,
        (+,Tuple{UInt32,UInt32}),
        (cli.get_global_id, Tuple{Int64}),
        (<=,Tuple{UInt32,UInt32}),
        (_prod,Tuple{Tuple{UInt32}}),
        (+,Tuple{Float32,Float32}),
        (broadcast_index,Tuple{cli.GlobalPointer{Float32},Tuple{UInt32},UInt32}),
        (broadcast_index,Tuple{Float32,Tuple{UInt32},UInt32}),
        (-,Tuple{UInt32,UInt32}),
        cli.GlobalPointer{Float32},
        typeof(+),
        Tuple{UInt32},
        Float32,
        (UInt32, Tuple{Int64}),
        Bool,
        typeof(_prod),
        typeof(broadcast_index)
    ]
    @test length(deps) == length(deps_test)
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
    long x2temp2;
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
@test target_source == source


function custom_index_test(x)
    x[1, 1]
end

source, method, name = Transpiler.kernel_source(custom_index_test, (typeof(1f0*I),))
source_compare = """// dependencies
// #custom_index_test
__constant int FUNC_INST_x2custom_index_test = 0;
typedef int x2custom_index_test; // empty type emitted as an int
// UniformScaling{Float32}
struct  __attribute__ ((packed)) TYPUniformScaling_float{
    float x4;
};
typedef struct TYPUniformScaling_float UniformScaling_float;

// Base.#getindex
__constant int FUNC_INST_Base12getindex = 0;
typedef int Base12getindex; // empty type emitted as an int
// Transpiler.CLIntrinsics.#cl_select
__constant int FUNC_INST_Transpiler1CLIntrinsics12cl_select = 0;
typedef int Transpiler1CLIntrinsics12cl_select; // empty type emitted as an int
// Base.Sort.#select
__constant int FUNC_INST_Base1Sort12select = 0;
typedef int Base1Sort12select; // empty type emitted as an int
// (select, Tuple{Float32,Float32,UInt32})
float select_9(float a, float b, uint c)
{
    if((bool)(c)){
        return a;
    };
    return b;
}
// (Transpiler.CLIntrinsics.cl_select, Tuple{Float32,Float32,Bool})
float cl_select_10(float a, float b, bool c)
{
    return select_9(a, b, (uint)(c));
}
// Symbol

// Base.#zero
__constant int FUNC_INST_Base12zero = 0;
typedef int Base12zero; // empty type emitted as an int
// Base.#oftype
__constant int FUNC_INST_Base12oftype = 0;
typedef int Base12oftype; // empty type emitted as an int
// (oftype, Tuple{Float32,Int64})
float oftype_11(float x, long c)
{
    return (float){c};
}
// (zero, Tuple{Float32})
float zero_6(float x)
{
    return oftype_11(x, 0);
}
// (getindex, Tuple{UniformScaling{Float32},Int64,Int64})
float getindex_7(UniformScaling_float J, long i, long j)
{
    return cl_select_10(J.λ, zero_6(J.λ), i == j);
}
// ########################
// Main inner function
// (custom_index_test, (UniformScaling{Float32},))
__kernel float custom_index_test_8(UniformScaling_float x)
{
    return getindex_7(x, 1, 1);
}
"""
@test source_compare == source

inner(i) = Float32(i) * 77f0
function ntuple_test(::Val{N}) where N
    ntuple(inner, Val{N})
end

source, method, name = Transpiler.kernel_source(ntuple_test, (Val{4},))
compare_source = """// dependencies
// #ntuple_test
__constant int FUNC_INST_x2ntuple_test = 0;
typedef int x2ntuple_test; // empty type emitted as an int
// Val{4}
typedef int Val_4; // empty type emitted as an int
// #inner
__constant int FUNC_INST_x2inner = 0;
typedef int x2inner; // empty type emitted as an int
// Type{Val{4}}
typedef int Type5Val5466; // placeholder type instance
__constant Type5Val5466 TYP_INST_Type5Val5466 = 0;

// Base.#ntuple
__constant int FUNC_INST_Base12ntuple = 0;
typedef int Base12ntuple; // empty type emitted as an int
// Any
typedef int Any; // placeholder type instance
__constant Any TYP_INST_Any = 0;

// (inner, Tuple{Int64})
float inner_14(long i)
{
    return (float)(i) * 77.0f;
}
// (ntuple, Tuple{#inner,Type{Val{4}}})
float4 ntuple_12(x2inner f, Type5Val5466 x2unused2)
{
    return (float4){inner_14(1), inner_14(2), inner_14(3), inner_14(4)};
}
// ########################
// Main inner function
// (ntuple_test, (Val{4},))
__kernel float4 ntuple_test_13(Val_4 x2unused2)
{
    return ntuple_12(FUNC_INST_x2inner, TYP_INST_Type5Val5466);
}
"""
@test compare_source == source
