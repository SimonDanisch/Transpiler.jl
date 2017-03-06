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

function test(a, b, c)
    gid = get_global_id(0) + 1
    c[gid] = a[gid] + b[gid]
    return
end

a = rand(Float32, 50_000)
b = rand(Float32, 50_000)
a_buff = cl.Buffer(Float32, ctx, (:r, :copy), hostbuf = a)
b_buff = cl.Buffer(Float32, ctx, (:r, :copy), hostbuf = b)
c_buff = cl.Buffer(Float32, ctx, :w, length(a))


program = ComputeProgram(test, (a_buff, b_buff, c_buff), ctx)

queue(program.program, size(a), nothing, a_buff, b_buff, c_buff)
r = cl.read(queue, c_buff)
if isapprox(norm(r - (a+b)), zero(Float32))
    info("Success!")
else
    error("Norm should be 0.0f")
end
