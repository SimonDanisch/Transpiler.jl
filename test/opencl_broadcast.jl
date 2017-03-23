using Transpiler
using OpenCL: cl
using Transpiler: CLTranspiler
import Transpiler.CLTranspiler.cli
import Transpiler.CLTranspiler.ComputeProgram
import Transpiler.CLTranspiler.cli.get_global_id
using Sugar, GeometryTypes, FileIO
import Sugar: ssavalue_name, ASTIO, get_slottypename, get_type, LazyMethod
import Sugar: getsource!, dependencies!, istype, isfunction, getfuncargs

function test{T}(a::T, b)
    x = sqrt(sin(a) * b) / T(10.0)
    y = T(33.0)x + cos(b)
    y * T(10.0)
end

device, ctx, queue = cl.create_compute_context()

a = rand(Float32, 50_000)
b = rand(Float32, 50_000)
device, ctx, queue = cl.create_compute_context()
a_buff = cl.CLArray(queue, a)
b_buff = cl.CLArray(queue, b)
c_buff = cl.CLArray(queue, similar(a))

import Transpiler.CLTranspiler: CLArray, CLMethod, CLIO
#Broadcast
Base.@propagate_inbounds broadcast_index(arg, shape, i) = arg
Base.@propagate_inbounds function broadcast_index{T, N}(
        arg::AbstractArray{T, N}, shape::NTuple{N, Integer}, i
    )
    @inbounds return arg[i]
end

_prod{T}(x::NTuple{1, T}) = x[1]
function _prod{N, T}(x::NTuple{N, T})
    x[1] * _prod(Base.tail(x))
end

for i = 0:10
    args = ntuple(x-> Symbol("arg_", x), i)
    fargs = ntuple(x-> :(broadcast_index($(args[x]), sz, i)), i)
    @eval begin
        function broadcast_kernel(A, f, sz, $(args...))
            i = get_global_id(0) + 1
            @inbounds if i <= _prod(sz)
                A[i] = f($(fargs...))
            end
            return
        end
    end
end
args = (cli.CLArray{Float32, 1}(), test, (length(a_buff),), (Val{true}(), Val{false}()), cli.CLArray{Float32, 1}(), 1f0)
args2 = (c_buff, test, (Int32(length(c_buff)),), a_buff, b_buff)
program = ComputeProgram(broadcast_kernel, args2, queue)
program(args2)
x = cl.to_host(c_buff)
