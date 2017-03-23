using Transpiler
using OpenCL: cl
import Transpiler.CLTranspiler.cli
const clt = Transpiler.CLTranspiler

function test{T}(a::T, b)
    x = sqrt(sin(a) * b) / T(10.0)
    y = T(33.0)x + cos(b)
    y * T(10.0)
end

function mapkernel(f, a, b, c)
    gid = cli.get_global_id(0) + 1
    c[gid] = f(a[gid], b[gid])
    return
end

a = rand(Float32, 50_000)
b = rand(Float32, 50_000)
device, ctx, queue = cl.create_compute_context()
a_buff = cl.CLArray(queue, a)
b_buff = cl.CLArray(queue, b)
c_buff = cl.CLArray(queue, similar(a))
args = (test, a_buff, b_buff, c_buff)

cl_mapkernel = clt.ComputeProgram(mapkernel, args, queue)

println(cl_mapkernel.source)
# call kernel. Accepts kw_args for global and local work size!
# but can also find them out automatically (in a super primitive way)

cl_mapkernel((test, a_buff, b_buff, c_buff))
r = cl.to_host(c_buff)
r2 = test.(a, b)
if all(isapprox.(r, r2))
    info("Success!")
else
    error("Norm should be 0.0f")
end
