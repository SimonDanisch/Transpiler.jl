using StaticArrays, Transpiler, Transpiler.CLTranspiler
import Transpiler.CLTranspiler: cli, CLMethod

const Vec = SVector
const Vec3f0 = SVector{3, Float32}
const Point3f0 = SVector{3, Float32}

function velocity_one_form(i, i2, psi, hbar)
    psi12  = psi[i[1],  i[2],  i[3]]
    psix12 = psi[i2[1], i[2],  i[3]]
    psiy12 = psi[i[1],  i2[2] ,i[3]]
    psiz12 = psi[i[1],  i[2],  i2[3]]
    psi1n = Vec(psix12[1], psiy12[1], psiz12[1])
    psi2n = Vec(psix12[2], psiy12[2], psiz12[2])
    angle.(
        conj(psi12[1]) * psi1n .+
        conj(psi12[2]) * psi2n
    ) * hbar
end

function velodiv(i, i2, velocity, res, ds)
    x, y, z = i
    ix, iy, iz = i2
    v1 = velocity[x, y,  z]
    v2 = Vec(
        velocity[ix, y,  z][1],
        velocity[x,  iy, z][2],
        velocity[x,  y, iz][3]
    )
    sum((v1 .- v2) .* ds)
end

function staggered_advect(p, args)
    velocity, dt, d, gridsize, res = args
    k1 = staggered_velocity(velocity, p, d, gridsize, res)

    k2 = p + k1 .* dt * 0.5f0
    k2 = staggered_velocity(velocity, k2, d, gridsize, res)

    k3 = p + k2 .* dt * 0.5f0
    k3 = staggered_velocity(velocity, k3, d, gridsize, res)

    k4 = p + k3 .* dt
    k4 = staggered_velocity(velocity, k4, d, gridsize, res)

    p .+ dt/6f0 .* (k1 .+ 2f0*k2 .+ 2f0*k3 .+ k4)
end

@inline function staggered_velocity(velocity, point, d, gs, res)
    p   = mod.(Vec(point), gs)
    i   = Vec{3, Int32}(floor.(p ./ d)) + Int32(1)
    ip  = mod.(i, res) + Int32(1)

    v0  = velocity[i[1], i[2], i[3]]

    pxp = velocity[ip[1], i[2], i[3]]
    pyp = velocity[i[1], ip[2], i[3]]
    pzp = velocity[i[1], i[2], ip[3]]

    vn = Vec3f0(
        velocity[i[1], ip[2], ip[3]][1],
        velocity[ip[1], i[2], ip[3]][2],
        velocity[ip[1], ip[2], i[3]][3]
    )
    pp  = Vec3f0(pyp[1], pxp[2], pxp[3])
    pp2 = Vec3f0(pzp[1], pzp[2], pyp[3])

    w   = p - (i - 1) .* d
    w1  = Vec3f0(w[3], w[3], w[2])
    w2  = Vec3f0(w[2], w[1], w[1])

    return Point3f0(
        (1f0 - w1) .* ((1f0 - w2) .* v0 + w2 .* pp) .+
        w1 .* ((1f0 - w2) .* pp2 + w2 .* vn)
    )
end
