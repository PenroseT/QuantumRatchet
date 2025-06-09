using LinearAlgebra
using NumericalIntegration

function vectorize_lindbladian(H::Array{ComplexF64}, γ::Float64, T::Float64)
    # Creates a Lindbladian with the thermal dissipator
    # Outputs L as matrix, and the unitary that diagonalizes the Hamiltonian

    Hdiag, U_H = eigen(H)
    Ham = diagm(Hdiag)
    λ = build_decayT(γ, Hdiag, T)
    λ = sqrt.(λ)
    Id = diagm(ones(size(H)[1], ))
    
    L = -1im*(kron(Id, Ham)-kron(transpose(Ham), Id))+kron(conj(λ), λ)-0.5*kron(Id, λ'*λ)-0.5*kron(transpose(λ'*λ), Id)
    
    return L, U_H
end

function unitary2R(ϕ1::Float64, ϕ2::Float64, θ::Float64)
    # 2x2 unitary that defines the measurement basis.
    # In order for the unitary to define the measurement
    # axis on the Bloch sphere ϕ1 is chosen to be pi/2
    # and ϕ2 is shifted by pi/2.
    
    U2 = zeros(ComplexF64, 2, 2)
    
    σ0 = [1 0; 0 1]
    σx = [0 1; 1 0]
    σy = [0 -1im; 1im 0]
    σz = [1 0; 0 -1]
    
    nx = sin(ϕ1)*cos(ϕ2)
    ny = sin(ϕ1)*sin(ϕ2)
    nz = cos(ϕ1)
    
    U2 = @. cos(θ/2)*σ0-1im*sin(θ/2)*(nx*σx+ny*σy+nz*σz)
   
    return U2
end

function measurement_projectors(ϕ1::Float64, ϕ2::Float64, θ::Float64, N::Int64, U_H::Array{ComplexF64})
    # The function takes parameters of the bond unitary and creates projectors for
    # a complete measurement in the bond basis. The output of the function is a
    # three-dimensional array whose z-axis slices are 2*N projectors that specify
    # the measurement.
    
    # The output is given in the initial (position) basis and in energy basis
    
    U2 = unitary2R(ϕ1, ϕ2, θ)
    
    Pjs = zeros(ComplexF64, 2*N, 2*N, 2*N)
    Pjsϵ = zeros(ComplexF64, 2*N, 2*N, 2*N)
     
    for i in 1:N
        Pjs[2*i-1:2*i, 2*i-1:2*i, 2*i-1] = [U2[1, 1]*conj(U2[1, 1]) U2[1, 1]*conj(U2[2, 1]); U2[2, 1]*conj(U2[1, 1]) U2[2, 1]*conj(U2[2, 1])]
        Pjs[2*i-1:2*i, 2*i-1:2*i, 2*i] = [U2[1, 2]*conj(U2[1, 2]) U2[1, 2]*conj(U2[2, 2]); U2[2, 2]*conj(U2[1, 2]) U2[2, 2]*conj(U2[2, 2])]
    end
    
    # Projectors in the energy eigenbasis:
    for i in 1:2*N
        Pjsϵ[:, :, i] = (U_H')*Pjs[:, :, i]*U_H
    end
    
    return Pjs, Pjsϵ
end

function measurement_lindbladian(H::Array{ComplexF64}, γ::Float64, T::Float64, θ::Float64, ϕ::Float64,
                                τm::Float64)                             
    # The function takes as input: Hamiltonian H, the decay matrix at a given decay rate γ and temperature T,
    # and a full set of projection operators constructed from the measured observable, at a measurement time τm.
    # The output is the Linbladian in the vectorized form and transition matrix from the position to
    # the energy eigenbasis used for basis transformations.
    
    N = size(H)[1]
    
    Hdiag, U_H = eigen(H)
    Ham = diagm(Hdiag)
    λ = build_decayT(γ, Hdiag, T)
    λ = sqrt.(λ)
    Id = diagm(ones(N, ))
    
    Pjs, Pjsϵ = measurement_projectors(π/2, ϕ+π/2, θ, Int(N÷2), U_H)
    Ls = jump_operators(λ)
    
    #The decay is not good (probably), decay channels should be separated
    
    L = -1im*(kron(Id, Ham)-kron(transpose(Ham), Id))
    
    for i in 1:size(Ls)[3]
        λ = Ls[:, :, i]
        L+=kron(conj(λ), λ)-0.5*kron(Id, λ'*λ)-0.5*kron(transpose(λ'*λ), Id)
    end
    
    for i in 1:size(Pjsϵ)[3]
        L.+=(1/τm)*(kron(conj(Pjsϵ[:,:, i]), Pjsϵ[:,:, i])-0.5*kron(Id, Pjsϵ[:,:, i])-0.5*kron(transpose(Pjsϵ[:,:, i]), Id))
    end
    
    return L, U_H
end

function build_decayT(γ::Float64, ens::Vector{}, T::Float64)
    # Constructs the thermal decay dissipator matrix lambda for at a given
    # decay rate gamma, temperature T, from the energy spectrum.
    
    N = length(ens)
    
    λmat = (γ/N)*ones(Float64, N, N)
    λmat[diagind(λmat)] = zeros(Float64, N)
    isapprox(0.0, T, atol=1e-7) && (T=1e-5)
    
    if isapprox(0.0, T, atol=1e-7)
        for i in 1:N
            for j in 1:i
                println("here")
                λmat[i, j]=0.0 
            end
        end    
    else
        for i in 1:N
            for j in 1:i
                λmat[i, j]*=exp(-(ens[i]-ens[j])/T)
            end
        end            
    end
    
    return λmat
end

function jump_operators(λ::Array{Float64})
    # Takes the lambda matrix and returns
    # N*(N-1) decay operators that constitute
    # a thermal decay proces
   
    N = size(λ)[1]
    
    Ls = zeros(Float64, N, N, N*(N-1))
    i = 1
    
    for j in eachindex([λ...])
        if (j-1)%(N+1)!=0
            _l = zeros(Float64, N*N, )
            _l[j] = λ[j]
            _lmat = reshape(_l, N, N)
            Ls[:, :, i] = _lmat
            i+=1
        end
    end
    
    return Ls
end

function diagonalize(L::Array{ComplexF64})
    # Diagonalizes the Linbladian and sorts eigenvectors
    # by the real part of the eigenvalue.
    # The last entry has a vanishing real part, making it a steady state.
    
    lvals, lvecs = eigen(L, sortby=(val->(real(val))))
    
    for i in 1:length(lvals)
        lvecs[:, i] = lvecs[:, i]/norm(lvecs[:, i])
    end
    
    return lvals, lvecs
end

function matricize_dm(ρ::Vector{ComplexF64})
    # Inverse operation of vectorization.

    ρmat = reshape(transpose(ρ), (Int(sqrt(length(ρ))), Int(sqrt(length(ρ)))))
    
    return ρmat
end

function rice_mele_hamiltonian(t1::Float64, t2::Float64, V::Float64, N::Int64)
    # Constructs the Rice-Mele Hamiltonian, given two hopping integrals t1 and t2,
    # and staggered potential V.
    
    H1 = zeros(ComplexF64, 2*N, 2*N)
    H2 = zeros(ComplexF64, 2*N, 2*N)
    H3 = zeros(ComplexF64, 2*N, 2*N)
    
    for i in 1:N
        H1[2*i-1, 2*i-1] = V
        H1[2*i, 2*i] = -V
        
        H2[2*i-1, 2*i] = -t1
        H2[2*i, 2*i-1] = -t1
        
        i != N && (H3[2*i, 2*i+1] = -t2)
        i != N && (H3[2*i+1, 2*i] = -t2)
    end    
    
    H3[1, 2*N] = -t2
    H3[2*N, 1] = -t2
    
    H = H1+H2+H3
    
    return H
end

function current(t1::Float64, t2::Float64, N::Int64)
    # The Hamiltonian current operator is constructed from the 
    # hopping integrals.
    
    j1 = zeros(ComplexF64, 2*N, 2*N)
    j2 = zeros(ComplexF64, 2*N, 2*N)
    
    for i in 1:N
        j1[2*i-1, 2*i] = -1.0im*t1
        j1[2*i, 2*i-1] = 1.0im*t1
        
        i != N && (j2[2*i, 2*i+1] = -1.0im*t2)
        i != N && (j2[2*i+1, 2*i] = 1.0im*t2)
    end    
    
    j2[1, 2*N] = 1.0im*t2
    j2[2*N, 1] = -1.0im*t2  
    
    j = j1 + j2
    
    return j
end

function steady_state(t1::Float64, t2::Float64, V::Float64, γ::Float64, T::Float64,
                      θ::Float64, ϕ::Float64, τm::Float64, N::Int64)
    # Calculates the steady state of the Lindbladian given
    # a Poissonian measurement scheme and thermal dissipation.
    
    H = rice_mele_hamiltonian(t1, t2, V, N)
    L, U_H = measurement_lindbladian(H, γ, T, θ, ϕ, τm)
    lvals, lvecs = diagonalize(L)
    ρst = matricize_dm(lvecs[:, end])
    ρst=ρst/tr(ρst)
    
    return ρst, U_H
end

function current_h_ss(t1::Float64, t2::Float64, V::Float64, γ::Float64, T::Float64,
                      θ::Float64, ϕ::Float64, τm::Float64, N::Int64)
    # Hamiltonian current expectation value in the steady state.
    
    ρst, U_H = steady_state(t1, t2, V, γ, T, θ, ϕ, τm, N)
    j = U_H'*current(t1, t2, N)*U_H
    cur = tr(j*ρst)
    
    return real(cur)
end

function measurement_current(θ::Float64, ϕ::Float64, N::Int64)
    # The measurement current operator is defined on the first bond. Full current
    # follows from the translational symmetry.
    
    jm = zeros(ComplexF64, 2*N, 2*N)
    
    mx = sin(θ)*cos(ϕ)
    my = sin(θ)*sin(ϕ)
    mz = cos(θ)
    
    jm[1:2, 1:2] = [1-mz^2 -mz*(mx-1im*my); -mz*(mx+1im*my) mz^2-1]
    
    return N*jm
end

function current_m_ss(t1::Float64, t2::Float64, V::Float64, γ::Float64, T::Float64,
                      θ::Float64, ϕ::Float64, τm::Float64, N::Int64)    
    # The measurement current expectation value in the steady state.
    
    ρst, U_H = steady_state(t1, t2, V, γ, T, θ, ϕ, τm, N)
    j = U_H'*measurement_current(θ, ϕ, N)*U_H
    
    curm = tr(j*ρst)/(2*τm)
    
    return real(curm)
end

function current_tot_ss(t1::Float64, t2::Float64, V::Float64, γ::Float64, T::Float64,
                      θ::Float64, ϕ::Float64, τm::Float64, N::Int64)
    # Total current expectation value in the steady state.
    
    ρst, U_H = steady_state(t1, t2, V, γ, T, θ, ϕ, τm, N)
    jh = U_H'*current(t1, t2, N)*U_H
    jm = U_H'*measurement_current(θ, ϕ, N)*U_H

    curh = tr(jh*ρst)
    curm = tr(jm*ρst)/(2*τm)

    return real(curh)+real(curm)
end

function entropy_ss(t1::Float64, t2::Float64, V::Float64, γ::Float64, T::Float64,
                      θ::Float64, ϕ::Float64, τm::Float64, N::Int64)
    # The von Neumann Entropy of the steady state density matrix.
    
    ρst, U_H = steady_state(t1, t2, V, γ, T, θ, ϕ, τm, N)
    j = U_H'*measurement_current(θ, ϕ, N)*U_H
    
    S = -tr(ρst*log(ρst))
    
    return real(S)
end

function renyi_entropy_ss(t1::Float64, t2::Float64, V::Float64, γ::Float64, T::Float64,
                      θ::Float64, ϕ::Float64, τm::Float64, N::Int64, n::Int64, S0_tol=1e-7)
    # Renyi entropies of the steady state density matrix.
    
    ρst, U_H = steady_state(t1, t2, V, γ, T, θ, ϕ, τm, N)
    j = U_H'*measurement_current(θ, ϕ, N)*U_H
    
    if n==0
        pops = real.(eigvals(ρst))
        S = log(count(i->(i>S0_tol), pops))
    elseif n==1
        S = -tr(ρst*log(ρst))
    elseif n>1 
        S = log(tr(ρst^n))/(1-n)
    elseif n==-1 
        pops = real.(eigvals(ρst))
        S = log(1/maximum(pops))
    end

    return real(S)
end

function infT_distance_ss(t1::Float64, t2::Float64, V::Float64, γ::Float64, T::Float64,
                      θ::Float64, ϕ::Float64, τm::Float64, N::Int64)
    # The distnace between the von Neumann entropy of the steady state density matrix
    # and the infinite temperature density matrix.
    
    ρst, U_H = steady_state(t1, t2, V, γ, T, θ, ϕ, τm, N)
    ρinf = zeros(Float64, 2*N, 2*N)
    ρinf[diagind(ρinf)]=1/(2*N)*ones(Float64, 2*N)
    
    D = 0.5*tr(sqrt((ρst'-ρinf')*(ρst-ρinf)))
    
    return real(D)
end


function dm_evolution_cur(t1::Float64, t2::Float64, V::Float64, γ::Float64, T::Float64,
                      θ::Float64, ϕ::Float64, τm::Float64, N::Int64, dt::Float64, time::Float64)
    # Uses exponential matrix map to evolve the density matrix for
    # a given array of times and calculates the expectation value of the 
    # Hamiltonian and measurement currents for every time.
    
    
    ts = [0.0:dt:time; ]
    curs_h = Array{Float64}(undef, length(ts))
    curs_m = Array{Float64}(undef, length(ts))
    H = rice_mele_hamiltonian(t1, t2, V, N)
    L, U_H = measurement_lindbladian(H, γ, T, θ, ϕ, τm)
    cur_h = current(t1, t2, N)
    cur_m = measurement_current(θ, ϕ, N)
    ρeq = U_H'*thermal_dm(H, T)*U_H
    ρeq = reshape(ρeq, (4*N^2, 1))
    
    for i in eachindex(ts)
        ρ = exp(ts[i]*L)*ρeq
        ρ = reshape(ρ, (2*N, 2*N))
        curs_h[i] = real(tr(cur_h*ρ))
        curs_m[i] = real(tr(cur_m*ρ))
    end
    
    return ts, curs_h, curs_m
end

function thermal_dm(H::Array{ComplexF64}, T::Float64)
    # Constructs a thermal density matrix
    # for a given Hamiltonian and temperature. 
    
    ρeq = exp(-H/T)
    ρeq .= ρeq/tr(ρeq)

    return ρeq
end

function steady_state_operators(H::Array{ComplexF64}, γ::Float64, T::Float64, θ::Float64, ϕ::Float64,
                                τm::Float64, t::Float64)
    # The steady state operators in the Floquet measurement scheme with the measurement period τm.
    
    N = size(H)[1]
    
    Hdiag, U_H = eigen(H)
    Ham = diagm(Hdiag)
    λ = build_decayT(γ, Hdiag, T)
    λ = sqrt.(λ)
    Id = diagm(ones(N, ))
    
    Pjs, Pjsϵ = measurement_projectors(π/2, ϕ+π/2, θ, Int(N÷2), U_H)
    Ls = jump_operators(λ)
    
    L = -1im*(kron(Id, Ham)-kron(transpose(Ham), Id))
    P = zeros(ComplexF64, N*N, N*N)
    one = kron(Id, Id)
    
    for i in 1:size(Ls)[3]
        λ = Ls[:, :, i]
        L+=kron(conj(λ), λ)-0.5*kron(Id, λ'*λ)-0.5*kron(transpose(λ'*λ), Id)
    end
    
    expL1 = exp(t*L)
    expL2 = exp((τm-t)*L)
    
    for i in 1:size(Pjsϵ)[3]
        P.+=kron(conj(Pjsϵ[:,:, i]), Pjsϵ[:,:, i])
    end
    
    return expL1, expL2, P, one, U_H
end

function floquet_steady_state(t1::Float64, t2::Float64, V::Float64, γ::Float64, T::Float64,
                      θ::Float64, ϕ::Float64, τm::Float64, t::Float64, N::Int64)
    # The steady-state density matrix in the Floquet measurement scheme. 
    
    H = rice_mele_hamiltonian(t1, t2, V, N)
    expL1, expL2, P, one, U_H = steady_state_operators(H, γ, T, θ, ϕ, τm, t)
    lvals, lvecs = diagonalize(expL2*P*expL1-one)
    ρst = matricize_dm(lvecs[:, end])
    ρst=ρst/tr(ρst)
    
    return ρst, U_H
end

function floquet_current_h_ss(t1::Float64, t2::Float64, V::Float64, γ::Float64, T::Float64,
                      θ::Float64, ϕ::Float64, τm::Float64, t::Float64, N::Int64)
    # The Hamiltonian current expectation value for specific time t in the Floquet steady state.
    
    ρst, U_H = floquet_steady_state(t1, t2, V, γ, T, θ, ϕ, τm, t, N)
    j = U_H'*current(t1, t2, N)*U_H
    cur = tr(j*ρst)
    
    return real(cur)
end

function floquet_integrated_current_h_ss(t1::Float64, t2::Float64, V::Float64, γ::Float64, T::Float64,
                                         θ::Float64, ϕ::Float64, τm::Float64, dt::Float64, N::Int64)
    # The integrated total Hamiltonian current over one cycle in the Floquet steady state.
    
    ts = [0.0:dt:τm; ]
    curs = Vector{Float64}(undef, length(ts))
    
    for i in eachindex(ts)
        curs[i] = floquet_current_h_ss(t1, t2, V, γ, T, θ, ϕ, τm, ts[i], N)
    end
    
    jtot =  integrate(ts, curs)/τm
    
    return jtot
end

function floquet_current_m_ss(t1::Float64, t2::Float64, V::Float64, γ::Float64, T::Float64,
                      θ::Float64, ϕ::Float64, τm::Float64, N::Int64)
    # The measurement current expectation value in the Floquet steady-state.
    
    ρst, U_H = floquet_steady_state(t1, t2, V, γ, T, θ, ϕ, τm, 0.0, N)
    j = U_H'*measurement_current(θ, ϕ, N)*U_H
    
    curm = tr(j*ρst)/(2*τm)
    
    return real(curm)
end

function floquet_current_tot_ss(t1::Float64, t2::Float64, V::Float64, γ::Float64, T::Float64,
                      θ::Float64, ϕ::Float64, τm::Float64, dt::Float64, N::Int64)
    # The total current expectation value in the Floquet steady state.
    
    curh = floquet_integrated_current_h_ss(t1, t2, V, γ, T, θ, ϕ, τm, dt, N)
    curm = floquet_current_m_ss(t1, t2, V, γ, T, θ, ϕ, τm, N)
    
    return curh + curm
end