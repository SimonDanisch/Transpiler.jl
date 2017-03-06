using StaticArrays
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
    i   = Vec{3, Int}(floor.(p ./ d)) + 1
    ip  = mod.(i, res) + 1

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
        (1 - w1) .* ((1 - w2) .* v0 + w2 .* pp) +
        w1 .* ((1 - w2) .* pp2 + w2 .* vn)
    )
end

using Sugar
import Sugar: @lazymethod, getsource!, dependencies!, getast!, isfunction, istype
import Sugar: LazyMethod, returntype

module Test
using Sugar, GeometryTypes, FileIO
using GLWindow, GLAbstraction, ModernGL, Reactive, GLFW, GeometryTypes
import Sugar: ssavalue_name, ASTIO, get_slottypename, get_type, LazyMethod

const GLMethod = LazyMethod{:GL}

include(Pkg.dir("Transpiler", "src", "clike", "opengl", "intrinsics.jl"))
include(Pkg.dir("Transpiler", "src", "clike", "opengl", "rewriting.jl"))
include(Pkg.dir("Transpiler", "src", "clike", "opengl", "printing.jl"))
end
using .Test
velocity = Array{Vec3f0, 3}
point = Point3f0
d = Vec3f0
gs = Vec{3, Int}
res = Vec{3, Int}
m = LazyMethod{:GL}((staggered_velocity, (velocity, point, d, gs, res)))
println(getsource!(m))




const PSI = Tuple{Complex64, Complex64}
i = Vec{3, Int}
i2 = Vec{3, Int}
psi = Array{PSI, 3}
hbar = Float32
m = LazyMethod{:GL}((velocity_one_form, (i, i2, psi, hbar)))
ast = getast!(m)
println(sprint() do io
    Test.show_unquoted(Test.GLSLIO(io, m), ast, 0, 0)
end)

for elem in dependencies!(m)
    if isfunction(elem)
        println(elem.signature)
        if !Sugar.isintrinsic(elem)
            println(sprint() do io
                ast = getast!(elem)
                Test.show_unquoted(Test.GLSLIO(io, elem), ast, 0, 0)
            end)
        end
    end
end

elem = LazyMethod{:GL}((broadcast, (typeof(+), StaticArrays.SVector{3,Complex{Float32}}, StaticArrays.SVector{3,Complex{Float32}})))
println(sprint() do io
    ast = getast!(elem)
    Test.show_unquoted(Test.GLSLIO(io, elem), ast, 0, 0)
end)

map(+, SVector{3,Float32}(1,1,1), SVector{3,Float32}(1,1,1))
@which map(+, SVector{3, Float32}(1, 1, 1), SVector{3,Float32}(1, 1, 1))
code_typed(map, map(typeof, (+, SVector{3, Float32}(1, 1, 1), SVector{3,Float32}(1, 1, 1))), optimize = false)


function iradon(A, angles) # input angles [rad]



  return I;
end
function radon(A,angles)

    angles = angles + pi/2;

    w,h = size(A);

    Nr_div2 = floor(sqrt(w*w+h*h)/2);

    Nr = Int64(Nr_div2*2+1);

    L = length(angles);

    sinogram = zeros(Nr,L);

    SIN = sin(angles);
    COS = cos(angles);

    R = linspace(-0.5,0.5,Nr)*Nr;

    RSIN = zeros(Nr,L);
    RCOS = zeros(Nr,L);

    w2 = w/2;
    h2 = h/2;

      for a=1:L
        for k=1:Nr
          RSIN[k,a]=R[k]*SIN[a];
          RCOS[k,a]=R[k]*COS[a];
        end
      end

    for a=1:L
        for k=1:Nr
          for l=1:Nr

              x = RCOS[k,a] - RSIN[l,a] + w2;
              y = RSIN[k,a] + RCOS[l,a] + h2;

              x1 = floor(x);
              x2 = ceil(x);
              y1 = floor(y);
              y2 = ceil(y);

              if !(x1 <= 0 || x2 > w || y1 <= 0 || y2 > h)

                # BILINEAR INTERPOLATION - STARTS
                # x1 y1 - btm left
                # x1 y2 - top left
                # x2 y1 - btm rght
                # x2 y2 - top rght
                f11 = A[Int64(x1),Int64(y1)];
                f21 = A[Int64(x1),Int64(y2)]; # that looks like, correct way
                f12 = A[Int64(x2),Int64(y1)];
                f22 = A[Int64(x2),Int64(y2)];
                  value = 0;
                if x2==x1 && y2==y1
                  value = f11;
                elseif x2==x1
                  value = 1/(y2-y1)*( f11*(y2-y) + f22*(y-y1) );
                elseif y2==y1
                  value = 1/(x2-x1)*( f11*(x2-x) + f22*(x-x1) );
                else
                  value = 1/(x2-x1)/(y2-y1)*( f11*(x2-x)*(y2-y) +
                                              f21*(x-x1)*(y2-y) +
                                              f12*(x2-x)*(y-y1) +
                                              f22*(x-x1)*(y-y1) );
                end
                # BILINEAR INTERPOLATION - ENDS

                sinogram[k,a] += value;

              end
          end
        end
    end
    return sinogram;
end
using Images, TestImages, ImageView

angles = (0:1:359)/360*2*pi; # [rad]

z = shepp_logan(1000; highContrast=true);
sinogram = radon(z,angles);
A = sinogram;
angles = angles + pi/2;
N, nAngles = size(A);
I = zeros(N, N); # reconstruction
x = linspace(-0.5,0.5,N)
filter = abs(linspace(-1, 1, N))

# FT domain filtering
for t = 1:length(angles)
    fhat = fftshift(fft(view(A, :, t)))
    A[:, t] = real(ifft(ifftshift(fhat .* filter)))
end

XCOS = zeros(N, length(angles));
XSIN = zeros(N, length(angles));
for k=1:N
  for a=1:length(angles)
    XCOS[k,a]=x[k]*cos(angles[a]);
    XSIN[k,a]=x[k]*sin(angles[a]);
  end
end

function test2{T}(N, nAngles, XCOS, XSIN, I, A::Matrix{T})
    @inbounds for n = 1:N, m = 1:N
        @simd for t = 1:nAngles
            r = XCOS[m, t] + XSIN[n, t]
            index = round(Int, (r / sqrt(T(2)) + T(0.5)) * N) # sqrt(2) magnified
            if index > 0 && index <= N
                I[m, n] += A[index, t]
            end
        end
    end
end
B = Float32.(A)
I = zeros(Float32, N, N)
XCOSf0 = Float32.(XCOS)
XSINf0 = Float32.(XSIN)
@profile test2(N, nAngles, XCOSf0, XSINf0, I, B)
