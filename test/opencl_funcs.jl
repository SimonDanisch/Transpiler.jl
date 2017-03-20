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

function mapkernel(f, a, b, c)
    gid = get_global_id(0) + 1
    c[gid] = f(a[gid], b[gid])
    return
end

a = rand(Float32, 50_000)
b = rand(Float32, 50_000)
a_buff = cl.Buffer(Float32, ctx, (:r, :copy), hostbuf = a)
b_buff = cl.Buffer(Float32, ctx, (:r, :copy), hostbuf = b)
c_buff = cl.Buffer(Float32, ctx, :w, length(a))


program = ComputeProgram(mapkernel, (test, a_buff, b_buff, c_buff), ctx)

queue(program.program, size(a), nothing, nothing, a_buff, b_buff, c_buff)

r = cl.read(queue, c_buff)
r2 = test.(a, b)
if all(isapprox.(r, r2))
    info("Success!")
else
    error("Norm should be 0.0f")
end
import Transpiler.CLTranspiler: CLArray, CLMethod, CLIO
m = CLMethod((mapkernel, (typeof(test), CLArray{Float32, 1}, CLArray{Float32, 1}, CLArray{Float32, 1})))
gio = CLIO(IOBuffer(), m)
for elem in dependencies!(m, true)
    if Sugar.istype(elem)
        println(elem.signature)
    end
end
x = []
m2 = CLMethod((mapkernel, (typeof(test), CLArray{Float32, 1}, CLArray{Float32, 1}, CLArray{Float32, 1})))
Sugar.ast_dependencies!(m2, Expr(:block, Sugar.getfuncargs(m)...))

#Broadcast
Base.@propagate_inbounds broadcast_index(::Val{false}, arg, shape, i) = arg
Base.@propagate_inbounds function broadcast_index{T, N}(
        ::Val{true}, arg::AbstractArray{T, N}, shape::NTuple{N, Integer}, i
    )
    @inbounds return arg[i]
end

_prod{T}(x::NTuple{1, T}) = x[1]
function _prod{N, T}(x::NTuple{N, T})
    x[1] * _prod(Base.tail(x))
end
@generated function broadcast_index{T, N}(::Val{true}, arg::AbstractArray{T, N}, shape, i)
    idx = []
    for i = 1:N
        push!(idx, :(ifelse(s[$i] < shape[$i], 1, idx[$i])))
    end
    expr = quote
        $(Expr(:meta, :inline, :propagate_inbounds))
        s = size(arg)
        idx = ind2sub(shape, i)
        @inbounds return arg[$(idx...)]
    end
end

for i = 0:10
    args = ntuple(x-> Symbol("arg_", x), i)
    fargs = ntuple(x-> :(broadcast_index(which[$x], $(args[x]), sz, i)), i)
    fargs2 = ntuple(x-> :(broadcast_index($(args[x]), sz, i)), i)
    @eval begin
        function broadcast_kernel(A, f, sz, which, $(args...))
            i = get_global_id(0) + 1
            @inbounds if i <= _prod(sz)
                A[i] = f($(fargs...))
            end
            nothing
        end
    end
end
args = (cli.CLArray{Float32, 1}(), test, (length(a_buff),), (Val{true}(), Val{false}()), cli.CLArray{Float32, 1}(), 1f0)
args2 = (a_buff, test, (length(a_buff),), (Val{true}(), Val{false}()), b_buff, 1f0)

@code_warntype(broadcast_kernel(args...))

program = ComputeProgram(broadcast_kernel, args2, ctx)
cli.clintrinsic(CLMethod(Float32))


src = """
float test(float arg, cl_int1 shape, int i)
{
    return arg;
}
"""
p = cl.build!(cl.Program(ctx, source = src), raise = false)
for (dev, status) in cl.info(p, :build_status)
    if status == cl.CL_BUILD_ERROR
        error(cl.info(p, :build_log)[dev])
    end
end
T = typeof( (Val{true}(), Val{false}()))


test{N, T}(::Type{NTuple{N, T}}) = T
test(T)

Transpiler.CLTranspiler.is_fixedsize_array(T)
