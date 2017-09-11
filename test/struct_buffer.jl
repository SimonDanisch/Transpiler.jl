using Transpiler, Sugar
import Transpiler: cli

Base.size(x::cli.DeviceArray) = x.size

function Base.getindex(x::cli.DeviceArray{T, N}, i::Vararg{Integer, N}) where {T, N}
    ilin = Transpiler.gpu_sub2ind(size(x), Cuint.(i))
    return x.ptr[ilin]
end

Transpiler.gpu_sub2ind(Cuint.((4, 4)), Cuint.((1, 2)))

function test(A)
    A[Cuint(1), Cuint(1)] + 0f0
    return
end

m = Transpiler.CLMethod((test, Tuple{cli.DeviceArray{Float32, 2}}))
Sugar.track_types!(m)
println(Sugar.getsource!(m))
Sugar.isintrinsic(m)
for elem in Sugar.dependencies!(m, true)
    if !Sugar.isintrinsic(elem)
        println(Sugar.getsource!(elem))
    end
end
