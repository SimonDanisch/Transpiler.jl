using Transpiler
using Base.Test
import Transpiler: cli, CLMethod
using Sugar: getsource!, dependencies!

function mapkernel(f, a, b, c)
    gid = cli.get_global_id(0) + Cuint(1)
    c[gid] = f(a[gid], b[gid])
    return
end

function test_source(target, result)
    source_equal = target == result
    if source_equal
        @test true
    else
        @test false
        println("source unequal:\ntarget:\n", target)
        println("result:\n", result)
    end
end
# empty caches
Transpiler.empty_caches!()

args = (typeof(+), cli.GlobalPointer{Float32}, cli.GlobalPointer{Float32}, cli.GlobalPointer{Float32})
cl_mapkernel = CLMethod((mapkernel, args))
source = Sugar.getsource!(cl_mapkernel)
mapsource = """void mapkernel_1(Base123 f, __global float *  a, __global float *  b, __global float *  c);
void mapkernel_1(Base123 f, __global float *  a, __global float *  b, __global float *  c)
{
    uint gid;
    gid = get_global_id(0) + (uint)(1);
    float _ssavalue_0;
    _ssavalue_0 = (a)[gid - 0x00000001] + (b)[gid - 0x00000001];
    (c)[gid - 0x00000001] = _ssavalue_0;
    return;
}"""

@testset "map kernel" begin
    test_source(mapsource, source)
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
broadcastsource = """void broadcast_kernel_5(__global float *  A, Base123 f, uint sz, __global float *  arg1, float arg2);
void broadcast_kernel_5(__global float *  A, Base123 f, uint sz, __global float *  arg1, float arg2)
{
    uint i;
    i = get_global_id(0) + (uint)(1);
    if(i <= _prod_2(sz)){
        float _ssavalue_0;
        _ssavalue_0 = broadcast_index_3(arg1, sz, i) + broadcast_index_4(arg2, sz, i);
        (A)[i - 0x00000001] = _ssavalue_0;
    };
    return;
}"""

@testset "broadcast kernel" begin
    test_source(broadcastsource, source)

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

@testset "for" begin
    decl = CLMethod((fortest, (Float32,)))
    source = Sugar.getsource!(decl)
    #TODO remove xxtempx4, which is unused now...
    target_source = """float fortest_6(float x);
    float fortest_6(float x)
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
    test_source(target_source, source)
end

function custom_index_test(x)
    x[1, 1]
end

@testset "custom getindex" begin
    source, method, name = Transpiler.kernel_source(custom_index_test, (typeof(1f0*I),))
    source_compare = """// Inbuilds
    typedef char JLBool;
    // dependencies
    // #custom_index_test
    __constant int FUNC_INST_x2custom_index_test = 0;
    typedef int x2custom_index_test; // empty type emitted as an int
    // UniformScaling{Float32}
    struct __attribute__((packed)) TYPUniformScaling_float{
        float __attribute__((aligned (4))) x4;
    };
    typedef struct TYPUniformScaling_float UniformScaling_float;

    // Base.#getindex
    __constant int FUNC_INST_Base12getindex = 0;
    typedef int Base12getindex; // empty type emitted as an int
    // Transpiler.CLIntrinsics.#cl_select
    __constant int FUNC_INST_Transpiler1CLIntrinsics12cl_select = 0;
    typedef int Transpiler1CLIntrinsics12cl_select; // empty type emitted as an int
    // (Transpiler.CLIntrinsics.cl_select, Tuple{Float32,Float32,Bool})
    float cl_select_9(float a, float b, JLBool c);
    float cl_select_9(float a, float b, JLBool c)
    {
        return select(b, a, (uint)(c));
    }
    // Symbol

    // Base.#zero
    __constant int FUNC_INST_Base12zero = 0;
    typedef int Base12zero; // empty type emitted as an int
    // Base.#oftype
    __constant int FUNC_INST_Base12oftype = 0;
    typedef int Base12oftype; // empty type emitted as an int
    // (oftype, Tuple{Float32,Int64})
    float oftype_10(float x, long c);
    float oftype_10(float x, long c)
    {
        return (float){c};
    }
    // (zero, Tuple{Float32})
    float zero_6(float x);
    float zero_6(float x)
    {
        return oftype_10(x, 0);
    }
    // (getindex, Tuple{UniformScaling{Float32},Int64,Int64})
    float getindex_7(UniformScaling_float J, long i, long j);
    float getindex_7(UniformScaling_float J, long i, long j)
    {
        return cl_select_9(J.λ, zero_6(J.λ), i == j);
    }
    // ########################
    // Main inner function
    // (custom_index_test, (UniformScaling{Float32},))
    __kernel float custom_index_test_8(UniformScaling_float x);
    float custom_index_test_8(UniformScaling_float x)
    {
        return getindex_7(x, 1, 1);
    }
    """
    test_source(source_compare, source)
end

inner(i) = Float32(i) * 77f0
function ntuple_test(::Val{N}) where N
    ntuple(inner, Val{N})
end
@testset "ntuple" begin
    source, method, name = Transpiler.kernel_source(ntuple_test, (Val{4},))
    compare_source = """// Inbuilds
    typedef char JLBool;
    // dependencies
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
    float inner_13(long i);
    float inner_13(long i)
    {
        return (float)(i) * 77.0f;
    }
    // (ntuple, Tuple{#inner,Type{Val{4}}})
    float4 ntuple_11(x2inner f, Type5Val5466 x2unused2);
    float4 ntuple_11(x2inner f, Type5Val5466 x2unused2)
    {
        return (float4){inner_13(1), inner_13(2), inner_13(3), inner_13(4)};
    }
    // ########################
    // Main inner function
    // (ntuple_test, (Val{4},))
    __kernel float4 ntuple_test_12(Val_4 x2unused2);
    float4 ntuple_test_12(Val_4 x2unused2)
    {
        return ntuple_11(FUNC_INST_x2inner, TYP_INST_Type5Val5466);
    }
    """
    test_source(compare_source, source)
end

function testifelse(a, b)
    ifelse(a == b, a, b)
end
@testset "ifelse" begin
    source, method, name = Transpiler.kernel_source(testifelse, (Int, Int))
    testsource = """// Inbuilds
    typedef char JLBool;
    // dependencies
    // #testifelse
    __constant int FUNC_INST_x2testifelse = 0;
    typedef int x2testifelse; // empty type emitted as an int
    // Transpiler.CLIntrinsics.#cl_select
    __constant int FUNC_INST_Transpiler1CLIntrinsics12cl_select = 0;
    typedef int Transpiler1CLIntrinsics12cl_select; // empty type emitted as an int
    // (Transpiler.CLIntrinsics.cl_select, Tuple{Int64,Int64,Bool})
    long cl_select_14(long a, long b, JLBool c);
    long cl_select_14(long a, long b, JLBool c)
    {
        return select(b, a, (ulong)(c));
    }
    // ########################
    // Main inner function
    // (testifelse, (Int64, Int64))
    __kernel long testifelse_15(long a, long b);
    long testifelse_15(long a, long b)
    {
        return cl_select_14(a, b, a == b);
    }
    """
    test_source(testsource, source)
end


function testfastmath(a::Complex64)
    @fastmath exp(a)
end
@testset "fastmath" begin
    source, method, name = Transpiler.kernel_source(testfastmath, (Complex64,))
    testsource = """// Inbuilds
    typedef char JLBool;
    // dependencies
    // #testfastmath
    __constant int FUNC_INST_x2testfastmath = 0;
    typedef int x2testfastmath; // empty type emitted as an int
    // Complex{Float32}
    struct __attribute__((packed)) TYPComplex_float{
        float __attribute__((aligned (4))) re;
        float __attribute__((aligned (4))) im;
    };
    typedef struct TYPComplex_float Complex_float;

    // Base.FastMath.#exp_fast
    __constant int FUNC_INST_Base1FastMath12exp_fast = 0;
    typedef int Base1FastMath12exp_fast; // empty type emitted as an int
    // Base.FastMath.#mul_fast
    __constant int FUNC_INST_Base1FastMath12mul_fast = 0;
    typedef int Base1FastMath12mul_fast; // empty type emitted as an int
    // Any
    typedef int Any; // placeholder type instance
    __constant Any TYP_INST_Any = 0;

    // Type{Float32}
    typedef int Type5Float326; // placeholder type instance
    __constant Type5Float326 TYP_INST_Type5Float326 = 0;

    // (Complex{Float32}, Tuple{Float32,Float32})
    Complex_float x7Complex_float8_17(float re, float im);
    Complex_float x7Complex_float8_17(float re, float im)
    {
        return (Complex_float){re, im};
    }
    // Base.#real
    __constant int FUNC_INST_Base12real = 0;
    typedef int Base12real; // empty type emitted as an int
    // Symbol

    // (real, Tuple{Complex{Float32}})
    float real_16(Complex_float z);
    float real_16(Complex_float z)
    {
        return z.re;
    }
    // Base.#imag
    __constant int FUNC_INST_Base12imag = 0;
    typedef int Base12imag; // empty type emitted as an int
    // (imag, Tuple{Complex{Float32}})
    float imag_16(Complex_float z);
    float imag_16(Complex_float z)
    {
        return z.im;
    }
    // (Base.FastMath.mul_fast, Tuple{Float32,Complex{Float32}})
    Complex_float mul_fast_18(float a, Complex_float y);
    Complex_float mul_fast_18(float a, Complex_float y)
    {
        return x7Complex_float8_17(a * real_16(y), a * imag_16(y));
    }
    // Base.FastMath.#cis_fast
    __constant int FUNC_INST_Base1FastMath12cis_fast = 0;
    typedef int Base1FastMath12cis_fast; // empty type emitted as an int
    // (Base.FastMath.cis_fast, Tuple{Float32})
    Complex_float cis_fast_6(float x);
    Complex_float cis_fast_6(float x)
    {
        return x7Complex_float8_17(cos(x), sin(x));
    }
    // (Base.FastMath.exp_fast, Tuple{Complex{Float32}})
    Complex_float exp_fast_16(Complex_float x);
    Complex_float exp_fast_16(Complex_float x)
    {
        return mul_fast_18(exp(real_16(x)), cis_fast_6(imag_16(x)));
    }
    // ########################
    // Main inner function
    // (testfastmath, (Complex{Float32},))
    __kernel Complex_float testfastmath_16(Complex_float a);
    Complex_float testfastmath_16(Complex_float a)
    {
        return exp_fast_16(a);
    }
    """
    test_source(testsource, source)
end
