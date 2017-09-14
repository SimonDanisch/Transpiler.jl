using Transpiler, Sugar
import Transpiler: cli



Transpiler.gpu_sub2ind(Cuint.((4, 4)), Cuint.((1, 2)))

function test(A)
    A[Cuint(1), Cuint(1)] + 0f0
    return
end
using GPU
m = Transpiler.CLMethod((test, Tuple{cli.DeviceArray{Float32, 2}}))
Sugar.track_types!(m)
println(Sugar.getsource!(m))
Sugar.isintrinsic(m)
for elem in Sugar.dependencies!(m, true)
    if !Sugar.isintrinsic(elem)
        println(Sugar.getsource!(elem))
    end
end

using CLArrays

x = CLArray(rand(Float32, 32, 32))
pointer(x)
GPUArrays.gpu_call(x, (pointer(x),)) do state, ptr
    arr = Transpiler.DeviceArray(ptr, (Cuint(32), Cuint(32)))
    arr[1, 1] = 22f0
    return
end

Array(x)[1,1]

x = Transpiler.CLMethod((test, Tuple{CLArrays.KernelState, Transpiler.DeviceArray{Float32, 2}}))
Sugar.track_types!(x)
ast = Sugar.getast!(x)
x.cache[:tracked_types]
yy = x.dependencies[4]
yy.cache[:tracked_types]
Sugar.getast!(yy)

for elem in Sugar.dependencies!(x, true)
    if Sugar.isfunction(elem) && elem.signature[1] == size
        println(elem.cache[:tracked_types])
    end
end

x2 = Transpiler.CLMethod((setindex!, Tuple{Transpiler.DeviceArray{Float32, 2}, Float32, Int, Int}))

ast = Sugar.getast!(x2)
Sugar.track_types!(x2)
Sugar.getsource!(x2) |> println
x2.cache[:tracked_types]
for elem in Sugar.dependencies!(x2, true)
    if Sugar.isfunction(elem) && elem.signature[1] == size
        println(elem.cache[:tracked_types])
    end
end



using OpenCL

device, ctx, queue = cl.create_compute_context()

# create opencl buffer objects
# copies to the device initiated when the kernel function is called
a_buff = cl.Buffer(Float32, ctx, :w, 5)
src = """
struct __attribute__ ((packed)) TDeviceArray{
    __global float * ptr;
    uint2 size;
};
typedef struct TDeviceArray DeviceArray;

__kernel void main(uint2 size, __global float * ptr){
    DeviceArray devarray;
    devarray = (DeviceArray){ptr, size};
    devarray.ptr[0] = 22.0;
}

"""
p = cl.Program(ctx, source=src) |> cl.build!
sum_kernel = cl.Kernel(p, "main")

# call the kernel object with global size set to the size our arrays
sum_kernel[queue, (1,)](Cuint.((2, 2)), a_buff)

# perform a blocking read of the result from the device
r = cl.read(queue, a_buff)
