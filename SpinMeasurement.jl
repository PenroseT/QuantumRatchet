using LinearAlgebra
using LaTeXStrings

function rho_two_rates(mx::Float64, my::Float64, mz::Float64,
                   hx::Float64, hz::Float64, γ1::Float64,
                   γ2::Float64, τ::Float64)
    
    Γ1 = γ1*τ
    Γ2 = γ2*τ
    ηx = hx*τ
    ηz = hz*τ
    η = sqrt(ηx^2+ηz^2)
    hnorm = sqrt(hx^2+hz^2)
    mdoth=(mx*hx+mz*hz)/hnorm
    
    det = 2*η^2*(1-mdoth^2)/Γ1+4*ηz^2+4*ηx^2*Γ2/Γ1+(1-Γ2/Γ1)*mz^2+Γ2/Γ1+2*Γ2*(1+mz^2)+4*Γ2^2+2*Γ2^2*(1-mz^2)/Γ1
    
    rhoxnum = -mx*mz-2*η*my*mdoth+2*Γ2*mx*mz-4*ηx*ηz
    rhoynum = -my*mz-2*η*mx*mdoth-2*Γ2*my*mz+2*ηx*(1-2*Γ2)
    rhoznum = -mz^2-2*Γ2*(1+mz^2)-4*(ηz^2+Γ2^2)
    
    rho = [rhoxnum/det, rhoynum/det, rhoznum/det]
    
    return rho
end

function rho_en_relaxation(mx::Float64, my::Float64, mz::Float64,
                       hx::Float64, hy::Float64, hz::Float64,
                       η::Float64, Γ::Float64)

    
    hnorm=sqrt(hx^2+hy^2+hz^2)
    hnormp=sqrt(hx^2+hy^2)
    mvech=(mx*hy-my*hx)/hnorm
    mdoth=(mx*hx+my*hy+mz*hz)/hnorm
    mperph=mz-mdoth*hz/hnorm
    
    det=4*η^2*(1-mdoth^2)/Γ+0.5*(1+0.5*Γ)*(1+Γ+(mdoth)^2)+4*η^2
    
    numx=-mdoth*(2*η*mvech+(1+0.5*Γ)*mperph)-hx*(4*η^2+(1+0.5*Γ)*(0.5*Γ+(mdoth)^2))/hnorm
    numy=mdoth*(2*η*mperph+(1+0.5*Γ)*(0.5*Γ+(mdoth)^2))*hnorm/hx
    numz=mdoth*(2*η*mvech-(1+0.5*Γ)*mperph)-hz*(4*η^2+(1+0.5*Γ)*(0.5*Γ+mdoth^2))/hnorm
    
    mx = numx/det
    my = numy/det
    mz = numz/det
    
    return [mx, my, mz]
end

function ss_matrix_weak_en(mx::Float64, my::Float64, mz::Float64,
                       hx::Float64, hy::Float64, hz::Float64,
                       γ0::Float64, τ::Float64, ξ::Float64)
    
    if ξ<0.0 || ξ>π/2
        throw(DomainError(ξ, "The parameter takes value between 0 and pi/2"))
    end
    
    hnorm=sqrt(hx^2+hy^2+hz^2)
    hnormp=sqrt(hx^2+hy^2)
    mvech=(mx*hy-my*hx)/hnorm
    mdoth=(mx*hx+my*hy+mz*hz)/hnorm
    mperph=mz-mdoth*hz/hnorm
    
    
    mxh = -hnorm*mperph/hnormp
    myh = -hnorm*mvech/hnormp
    mzh = mdoth
    
    Γ=γ0*τ
    η=hnorm*τ
    s=(sin(ξ))^2
    
    det=4*s*η^2*(1-mdoth^2)/Γ+0.5*s^2*(1+mdoth^2)+0.75*Γ*s*(1+mdoth^2/3)+4*η^2+Γ^2/4
    
    numx = s*(2*η*myh*mzh-(s+Γ/2)*mxh*mzh)
    numy = s*(-2*η*mxh*mzh-(s+Γ/2)*myh*mzh)
    numz = -(s+Γ/2)*(Γ/2+mzh^2*s)-4*η^2

    rhox = numx/det
    rhoy = numy/det
    rhoz = numz/det
    
    return [rhox, rhoy, rhoz]
end

function ss_matrix_weak(mx::Float64, my::Float64, mz::Float64,
                       hx::Float64, hy::Float64, hz::Float64,
                       γ0::Float64, τ::Float64, ξ::Float64)
    
    if ξ<0.0 || ξ>π/2
        throw(DomainError(ξ, "The parameter takes value between 0 and pi/2"))
    end
    
    hnorm=sqrt(hx^2+hy^2+hz^2)
    hnormp=sqrt(hx^2+hy^2)
    mvech=(mx*hy-my*hx)/hnorm
    mdoth=(mx*hx+my*hy+mz*hz)/hnorm
    mperph=mz-mdoth*hz/hnorm
    
    
    mxh = -hnorm*mperph/hnormp
    myh = -hnorm*mvech/hnormp
    mzh = mdoth
    
    Γ=γ0*τ
    η=hnorm*τ
    s=(sin(ξ))^2
    
    det=4*s*η^2*(1-mdoth^2)/Γ+0.5*s^2*(1+mdoth^2)+0.75*Γ*s*(1+mdoth^2/3)+4*η^2+Γ^2/4
    
    numx=-mdoth*(2*η*s*mvech-s*(s+0.5*Γ)*mperph)-hx*(4*η^2+(s+0.5*Γ)*(0.5*Γ+s*(mdoth)^2))/hnorm
    numy=mdoth*(2*η*s*mperph+s*(s+0.5*Γ)*mvech)*hnorm/hx
    numz=mdoth*(2*η*s*mvech-s*(s+0.5*Γ)*mperph)-hz*(4*η^2+(s+0.5*Γ)*(0.5*Γ+s*mdoth^2))/hnorm
     

    rhox = numx/det
    rhoy = numy/det
    rhoz = numz/det
    
    return [rhox, rhoy, rhoz]
end

function entropy_weak(mx::Float64, my::Float64, mz::Float64,
                       hx::Float64, hy::Float64, hz::Float64,
                       γ0::Float64, τ::Float64, ξ::Float64)
    
    rho=ss_matrix_weak(mx, my, mz, hx, hy, hz, γ0, τ, ξ)
    rhonorm = sqrt(rho[1]^2+rho[2]^2+rho[3]^2)
    
    S = -0.5*(1+rhonorm)*log(0.5+0.5*rhonorm)-0.5*(1-rhonorm)*log(0.5-0.5*rhonorm)
   
    return S
end