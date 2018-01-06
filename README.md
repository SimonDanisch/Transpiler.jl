# Transpiler

[![Build Status](https://travis-ci.org/SimonDanisch/Transpiler.jl.svg?branch=master)](https://travis-ci.org/SimonDanisch/Transpiler.jl)

[![Coverage Status](https://coveralls.io/repos/SimonDanisch/Transpiler.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/SimonDanisch/Transpiler.jl?branch=master)

[![codecov.io](http://codecov.io/github/SimonDanisch/Transpiler.jl/coverage.svg?branch=master)](http://codecov.io/github/SimonDanisch/Transpiler.jl?branch=master)


Tools for working with Julia's typed AST and emiting code for other statically compiled languages.

Transpiling is not the optimal way to emit code for e.g. OpenCL or OpenGL, but it's a nice way to integrate simple user defined Julia functions into a larger framework (e.g. for [GPUArrays](https://github.com/JuliaGPU/GPUArrays.jl/)).
The better appraoch is via LLVM, like [CUDAnative](https://github.com/JuliaGPU/CUDAnative.jl/), and using SPIR-V for OpenCL and Vulkan.
Right now it's a nice adhoc solution to get our Julia -> GPU compilation efforts started and the transpilation code is much more appraochable for Julia programmers then diving into the world of LLVM + Julia internals.
Also, the tools developed for this packages (e.g. [Sugar](https://github.com/SimonDanisch/Sugar.jl) and [Matcha](https://github.com/SimonDanisch/Matcha.jl)) offer a lot of functionality needed for static linting and introspection into Julia's typed AST's.
Another option is to use Julia itself as a transpilation target and implement macros from a typed AST, allowing to do more powerful transformations.

### Installation

```Julia
Pkg.add("Transpiler")
```

### Example

```Julia
using Transpiler: kernel_source
import Transpiler: cli
using OpenCL

function test(a::T, b) where {T}
    x = sqrt(sin(a) * b) / T(10.0)
    y = T(33.0)x + cos(b)
    y * T(10.0)
end

function mapkernel(f, a, b, c)
    gid = cli.get_global_id(0) + 1
    c[gid] = f(a[gid], b[gid])
    return
end

# types of mapkernel arguments
T = cli.GlobalPointer{Float32}
argtypes = (typeof(test), T, T, T)
src, method, fname = kernel_source(mapkernel, argtypes)

println(src)

# setup OpenCL buffers, queue and context
a = rand(Float32, 50_000)
b = rand(Float32, 50_000)
device, ctx, queue = cl.create_compute_context()
a_buff = cl.Buffer(Float32, ctx, (:r, :copy), hostbuf=a)
b_buff = cl.Buffer(Float32, ctx, (:r, :copy), hostbuf=b)
c_buff = cl.Buffer(Float32, ctx, :w, length(a))

# compile kernel
p = cl.Program(ctx, source=src) |> cl.build!
k = cl.Kernel(p, fname)

# call kernel. Accepts kw_args for global and local work size!
# but can also find them out automatically (in a super primitive way)
queue(k, size(a), nothing, test, a_buff, b_buff, c_buff)

r = cl.read(queue, c_buff)

if r ≈ test.(a, b)
    info("Success!")
else
    error("⁉")
end
```

Output:

```OpenCL
// dependant type declarations
typedef struct {
float empty; // structs can't be empty
}_1test;

// dependant function declarations
float test_8633297058295171728(float a, float b)
{
    float y;
    float x;
    x = sqrt(sin(a) * b) / (float)(10.0);
    y = (float)(33.0) * x + cos(b);
    return y * (float)(10.0);
}
// Main inner function
__kernel void mapkernel_5672850724456951104(__global const _1test *f, __global float * a, __global float * b, __global float * c)
{
    int gid;
    gid = get_global_id(0) + 1;
    float _ssavalue_0;
    _ssavalue_0 = test_8633297058295171728(a[gid - 1], b[gid - 1]);
    c[gid - 1] = _ssavalue_0;
    ;
}
```

# TODO / Common issues

* compiling constructor code
* Not sure how to transpile `Core._apply`
* passing around types and constructing them
* better error handling / logging
