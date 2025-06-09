using LinearAlgebra
using SparseArrays
using SpecialFunctions: gamma, gamma_inc
using LsqFit
using Plots
using FFTW
using Combinatorics
pyplot()

######################################################################################################
################################### Object Construction ##############################################
######################################################################################################


struct momenta
    size::Int64
    vector::Vector{Float64}        
    function momenta(N::Int64)
       new(N, (2*π/N)*[0:1:N-1; ]) 
    end
end

struct momenta_trunc
    #Momenta that go from 0 to 2p/n
    size::Int64
    vector::Vector{Float64}        
    function momenta_trunc(N::Int64, n::Int64)
       new(N, (2*π/(n*N))*[0:1:N-1; ]) 
    end
end


function make(matrices::Vector, args::Tuple, N::Int64)
    # this function should take a given set of matrices
    # multiply them with args, sum them up and return
    # the final matrix
    
    H = zeros(ComplexF64, N, N)
    
    for i in eachindex(args)
        H += args[i]*matrices[i]
    end
    return H
end

struct observable
    size::Int64
    shape::Tuple{Int64, Int64}
    args::Tuple
    matrix::Array{ComplexF64}
    vals::Vector{Float64}
    vectors::Array{ComplexF64}
    function observable(matrices::Vector,
                    args::Tuple, make)
        #The observable constructor
        
        N = size(matrices[1])[1]
        for matrix in matrices
            @assert size(matrix)[1]==N "The matrices don't have the same size"
        end
        
        @assert length(args)==length(matrices) "The number of matrices doesn't match
                                                the number of arguments"
        
        matrix = make(matrices, args, N)
        vals, vectors = eigen(matrix)
    
        for val in vals
           @assert isapprox(imag(val), 0.0, atol=1e-7) "The observable is not Hermitian" 
        end
        new(N, (N, N), args, matrix, real.(vals), vectors)
    end
end

#----------- Density matrices --------------#

struct density_matrix
    size::Int64
    shape::Tuple{Int64, Int64}
    matrix::Array{ComplexF64}
    function density_matrix(ρ::Array{ComplexF64})
        N = size(ρ)[1]
        new(N, (N, N), ρ)
    end
end

struct dm_initial
    size::Int64
    shape::Tuple{Int64, Int64}
    matrix::Array{ComplexF64}
    function dm_initial(N::Int64, n::Int64)
        @assert n<=N "The eigenstate number is larger than the Hspace dimension"
        new(N, (N, N), sparse([n], [n], [1.0], N, N)) 
    end
end

#------------ Two Band Model --------------# 

struct dm_initial_pos
    size::Int64
    shape::Tuple{Int64, Int64}
    matrix::Array{ComplexF64}
    function dm_initial_pos(ks::momenta, n::Int64)
        n>ks.size && error("Starting position is larger than system size")
        n<1 && error("Starting position has to be a positive integer")
        
        Ψ0 = [i%2==0 ? 0.0 : exp(-1im*ks.vector[Int(i÷2)+1])/sqrt(ks.size) for i in 1:2*ks.size]
        
        new(2*ks.size, (2*ks.size, 2*ks.size), Ψ0*Ψ0')
    end
end

struct bond_unitary
    size::Int64
    shape::Tuple{Int64, Int64}
    matrix::Array{ComplexF64}
    function bond_unitary(N::Int64)
        U = zeros(ComplexF64, 2*N, 2*N)    
        for i in 1:N
            U[2*i-1, 2*i-1] = 1/sqrt(2) 
            U[2*i-1, 2*i] = 1/sqrt(2)
            U[2*i, 2*i-1] = 1im/sqrt(2)
            U[2*i, 2*i] =  -1im/sqrt(2)
        end
        new(2*N, (2*N, 2*N), U)
    end
end

function unitary2(θ::Float64, ϕ1::Float64, ϕ2::Float64)
    
    U2 = zeros(ComplexF64, 2, 2)
    
    U2[1, 1] = exp(1im*ϕ1)*cos(θ)
    U2[1, 2] = exp(1im*ϕ2)*sin(θ)
    U2[2, 1] = -exp(-1im*ϕ2)*sin(θ)
    U2[2, 2] = exp(-1im*ϕ1)*cos(θ)
   
    return U2
end

function unitary2R(ϕ1::Float64, ϕ2::Float64, θ::Float64)
    
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

function bond_unitary_Ux(x::observable, U2::Array{ComplexF64})
    # This function creates an observable
    # whose projectors correspond to
    # applying the unitary to the bond
    # and measuring either of two sites

    N = Int(x.size ÷2)
    @assert size(U2)==(2, 2) "The input unitary has to be 2x2"
    
    U = zeros(ComplexF64, 2*N, 2*N)
    
    for i in 1:N
        U[2*i-1:2*i, 2*i-1:2*i] = U2
    end
    
    Ux = observable([U*x.matrix*U'], (1.0, ), make)
    
    return Ux
end


struct two_bond_unitary
    size::Int64
    shape::Tuple{Int64, Int64}
    matrix::Array{ComplexF64}
    function two_bond_unitary(N::Int64)
        @assert rem(N, 2)==0 "System has to have 4*k sites to allow the fragmentation" 
        U = zeros(ComplexF64, 2*N, 2*N)   
        U4 = 1/2 .* [1im -1im -1im 1im; 1 1 -1 -1; 1im -1im 1im -1im; 1 1 1 1] 
        for i in 1:Int(N÷2)
            U[4*i-3:4*i, 4*i-3:4*i] = U4
        end
        new(2*N, (2*N, 2*N), U)
    end
end

function initial_pos(ks::momenta, n::Int64)
    #Position eigenstate initial constructor
    
    n>ks.size && error("Starting position is larger than system size")
    n<1 && error("Starting position has to be a positive integer")
    
    Ψ0 = [i%2==0 ? 0.0+0.0im : exp(-1im*ks.vector[Int(i÷2)+1])/sqrt(ks.size) for i in 1:2*ks.size]
    ρmat = Ψ0*Ψ0'
    
    ρ = density_matrix(ρmat::Array{ComplexF64})
    
    return ρ
end

function initial_energy(N::Int64, n::Int64)
    #Position eigenstate initial constructor
    
    n>2*N && error("Starting energy is larger than Hilbert space size")
    n<1 && error("Energy index has to be a positive integer")
    
    Ψ0 = [i==n ? 1.0+0.0im : 0.0+0.0im for i in 1:N]
    ρmat = Ψ0*Ψ0'
    
    ρ = density_matrix(ρmat::Array{ComplexF64})
    return ρ
end

function initial_en(ks::momenta, n::Int64)

    n>2*ks.size && error("Energy index is larger than the Hilbert space dimension")
    n<1 && error("Energy index has to be a positive integer")
    
    Ψ0 = zeros(ComplexF64, 2*ks.size)
    
    Ψ0 = [i==n ? (i=1.0+0.0im) : (i=0.0+0.0im) for i in 1:2*ks.size]
    ρmat = Ψ0*Ψ0'

    ρ = density_matrix(ρmat::Array{ComplexF64})

    return ρ 
end

#------------ Three Band Model --------------# 

function initial_pos_three(ks::momenta_trunc, n::Int64)

    n>ks.size && error("Starting position is larger than system size")
    n<1 && error("Starting position has to be a positive integer")
    
    Ψ0 = zeros(ComplexF64, 3*ks.size)
    
    for i in 1:ks.size
        Ψ0[3*i-2]=exp(-1im*(3*i-2)*ks.vector[i])/sqrt(ks.size)
    end
    
    ρmat = Ψ0*Ψ0'

    ρ = density_matrix(ρmat::Array{ComplexF64})

    return ρ 
end

function initial_en_three(ks::momenta_trunc, n::Int64)

    n>3*ks.size && error("Energy index is larger than the Hilbert space dimension")
    n<1 && error("Energy index has to be a positive integer")
    
    Ψ0 = zeros(ComplexF64, 3*ks.size)
    
    Ψ0 = [i==n ? (i=1.0+0.0im) : (i=0.0+0.0im) for i in 1:3*ks.size]
    ρmat = Ψ0*Ψ0'

    ρ = density_matrix(ρmat::Array{ComplexF64})

    return ρ 
end

#----------- Hamiltonian --------------#

struct Hamiltonian
    size::Int64
    shape::Tuple{Int64, Int64}
    args::Tuple
    matrix::Array{ComplexF64}
    ens::Vector{Float64}
    vectors::Array{ComplexF64}
    function Hamiltonian(matrices::Vector,
                         args::Tuple, make)
        
        #The Hamiltonian constructor takes arguments,
        #matrices and constructor function as inputs
        #and creates the Hamiltonian operator
        
        N = size(matrices[1])[1]
        for matrix in matrices
            @assert size(matrix)[1]==N "The matrices don't have the same size"
        end 
        
        @assert length(args)==length(matrices) "The number of matrices doesn't match
                                                the number of arguments"
        
        matrix = make(matrices, args, N)
        ens, vectors = eigen(matrix)
    
        for en in ens
           @assert isapprox(imag(en), 0.0, atol=1e-7) "The Hamiltonian is not Hermitian" 
        end
        new(N, (N, N), args, matrix, real.(ens), vectors)
    end
end 

#=====================================================================#
#-------------------------- Two Band Model ---------------------------#
#=====================================================================#


struct overlap_integral
    size::Int64
    vector::Vector{ComplexF64}
    function overlap_integral(t1::Float64, t2::Float64, ks::momenta)
        new(ks.size, @. t1+t2*exp(-1im*ks.vector))
    end
end

struct band
    size::Int64
    vector::Vector{Float64}
    function band(t1::Float64, t2::Float64, V::Float64, ks::momenta)
        ens = [sqrt(V^2/4+abs(t1+t2*exp(-1im*ks.vector[i]))^2) for i in 1:ks.size]
        new(ks.size, ens)
    end
end

struct current_integral
    size::Int64
    vector::Vector{ComplexF64}
    function current_integral(t1::Float64, t2::Float64, ks::momenta)
        new(ks.size, @. 1im*(t1-t2*exp(-1im*ks.vector)))
    end
end

function ratchet_build_current(t1, t2, N)
    # Current builder for two band model
    
    ks = momenta(N)
    c1, c2 = zeros(ComplexF64, 2*N, 2*N),
             zeros(ComplexF64, 2*N, 2*N)
                    
    for i in eachindex(ks.vector)
       c1[2*i-1, 2*i] = 1.0im
       c1[2*i, 2*i-1] = -1.0im
       c2[2*i-1, 2*i] = -1.0im*exp(-1im*ks.vector[i])
       c2[2*i, 2*i-1] = 1.0im*exp(1im*ks.vector[i])
    end

    args = (t1, t2)
    
    return [c1, c2], args
end


function ratchet_build(t1, t2, V, N)
    # Hamiltonian builder for two band model
    
    ks = momenta(N)
    h1, h2, h3 = zeros(ComplexF64, 2*N, 2*N),
                 zeros(ComplexF64, 2*N, 2*N),
                 zeros(ComplexF64, 2*N, 2*N)
                    
    for i in eachindex(ks.vector)
       h1[2*i-1, 2*i] = 1.0
       h1[2*i, 2*i-1] = 1.0
       h2[2*i-1, 2*i] = 1.0*exp(-1im*ks.vector[i])
       h2[2*i, 2*i-1] = 1.0*exp(1im*ks.vector[i])
       h3[2*i-1, 2*i-1] = 1.0/2
       h3[2*i, 2*i] = -1.0/2
    end

    args = (t1, t2, V)
    
    return [h1, h2, h3], args
end

function build_position(N)
    
    ks = momenta(N)
    xmat = zeros(ComplexF64, 2*N, 2*N)
    
    for i in 1:N
        xmat[2*i-1, 2*i-1] = N/2
        xmat[2*i, 2*i] = (N+1)/2
    end

    for i in 1:N
        for j in i+1:N
            rij = exp(-1im*(ks.vector[i]-ks.vector[j]))
            xmat[2*i-1, 2*j-1] = (4*rij^2-2*N*rij*(1-rij))/(N*(1-rij)^2)
            xmat[2*i, 2*j] = (4*rij^2-2*N*rij*(1-rij))/(N*(1-rij)^2)
        end
    end
    
    xmat = xmat+xmat'
    
    x = observable([xmat], (1.0, ), make)
    
    return x
end

function build_decayA(γ::Float64, N::Int64)
    
    λmat = zeros(Float64, 2*N, 2*N)
    
    for i in 1:2*N-1
        λmat[i, i+1:end] = γ.*ones(2*N-i)
    end

    for i in 1:N-1
        λmat[2*i+1, 2*i] = γ
    end
    
    if N%2==0
        λmat[N+1, N]=0.0
    end
    
    λ = λmatrix(λmat)
    return λ
end

function build_decayB(γ::Float64, N::Int64)
    
    λmat = zeros(Float64, 2*N, 2*N)
    
    for i in 1:2*N-1
        λmat[i, i+1] = γ
        i != 2*N-1 && (λmat[i, i+2] = γ)
    end  

    for i in 1:N-1
        λmat[2*i+1, 2*i] = γ
    end
    
    for i in 2:N-1
        λmat[2*i-2, 2*i+1] = γ
    end
    
    if N%2==0
        λmat[N+1, N]=0.0
        λmat[N-1, N+1]=0.0
        λmat[N-2, N+1]=0.0
        λmat[N, N+2]=0.0
        λmat[N, N+3]=0.0
    end
    
    λ = λmatrix(λmat)
    
    return λ
end

function build_decayC(γ::Float64, N::Int64)
    
    λmat = zeros(Float64, 2*N, 2*N)
    
    for i in 1:2*N-1
        λmat[i, i+1] = γ
        i != 2*N-1 && (λmat[i, i+2] = γ)
    end

    for i in 1:N-1
        λmat[2*i, 2*i+1] = 0.0
    end
    
    for i in 2:N-1
        λmat[2*i-2, 2*i+1] = γ
    end
    
    if N%2==0
        λmat[N+1, N]=0.0
        λmat[N-1, N+1]=0.0
        λmat[N-2, N+1]=0.0
        λmat[N, N+2]=0.0
        λmat[N, N+3]=0.0
        λmat[N, N+1]=γ
    end
    
    λ = λmatrix(λmat)
    return λ
end

function build_decayT(γ::Float64, ens::Vector{Float64}, T::Float64)
    
    
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
    
    
    λ = λmatrix(λmat)
    
    return λ
end

function build_objects_2band(t1::Float64, t2::Float64, V::Float64,
                             γ::Float64, N::Int64, n::Int64; decay="A", temp=1.0, en=false)

    ks = momenta(N)
    
    if en==false
        ρ0 = initial_pos(ks, n)
    elseif en==true
        ρ0 = initial_en(ks, n)
    else
        error("en can be only true or false")
    end
    
    # Builds the Hamiltonian    
    hs, hargs = ratchet_build(t1, t2, V, N)
    H = Hamiltonian(hs, hargs, make)

    # Builds the curent and the current
    # projector structs
    cs, cargs = ratchet_build_current(t1, t2, N)
    J = observable(cs, cargs, make)
    
    if decay=="A"
        λ = build_decayA(γ, N)
    elseif decay=="B"
        λ = build_decayB(γ, N)
    elseif decay=="C"
        λ = build_decayC(γ, N)
    elseif decay=="T"
        λ = build_decayT(γ, H.ens, temp)
    else
        error("Decay can only be A, B, or C.")
    end
        
    x  = build_position(N)
   
    return ρ0, H, x, J, λ
end

function build_squares_2band(t1::Float64, t2::Float64, N::Int64)
    
    cs, cargs = ratchet_build_current(t1, t2, N)
    J = observable(cs, cargs, make)
    J2 = observable([J.matrix*J.matrix], (1.0, ), make)
    #x = build_position(N)
    #x2 = observable([x.matrix*x.matrix], (1.0, ), make)
    
    return J2
end

function build_occupation(N; site="a")
   
    siteindex = Dict("a"=>1, "b"=>0) 
    
    OccM = zeros(Float64, 2*N, 2*N)
    
    for i in 1:N
        OccM[2*i-siteindex[site], 2*i-siteindex[site]] = 1.0
    end
    
    Occ = observable([OccM], (1.0, ), make)
    
    return Occ
end


############ Building 2-band objects in coordinate space ##############


function hamiltonian_pos(t1::Float64, t2::Float64, V::Float64, N::Int64)
    
    H1 = zeros(ComplexF64, 2*N, 2*N)
    H2 = zeros(ComplexF64, 2*N, 2*N)
    H3 = zeros(ComplexF64, 2*N, 2*N)
    
    for i in 1:N
        H1[2*i-1, 2*i-1] = 1.0
        H1[2*i, 2*i] = -1.0
        
        H2[2*i-1, 2*i] = 1.0
        H2[2*i, 2*i-1] = 1.0
        
        i != N && (H3[2*i, 2*i+1] = 1.0)
        i != N && (H3[2*i+1, 2*i] = 1.0)
    end    
    
    H3[1, 2*N] = 1.0
    H3[2*N, 1] = 1.0
    
    H = Hamiltonian([H1, H2, H3], (V/2, t1, t2), make)
    
    return H
end

function current_pos(t1::Float64, t2::Float64, N::Int64)
    
    j1 = zeros(ComplexF64, 2*N, 2*N)
    j2 = zeros(ComplexF64, 2*N, 2*N)
    
    for i in 1:N
        j1[2*i-1, 2*i] = 1.0im
        j1[2*i, 2*i-1] = -1.0im
        
        i != N && (j2[2*i, 2*i+1] = 1.0im)
        i != N && (j2[2*i+1, 2*i] = -1.0im)
    end    
    
    j2[1, 2*N] = -1.0im
    j2[2*N, 1] = 1.0im  
    
    j = observable([j1, j2], (t1, t2), make)
    
    return j
end

function position_pos(N::Int64)
   
    xmat = zeros(ComplexF64, 2*N, 2*N)
    
    for i in 1:2*N
        xmat[i, i] = i
    end
    
    x = observable([xmat], (1.0, ), make)
    
    return x
end

function build_objects_2band_position(t1::Float64, t2::Float64, V::Float64,
                                      γ::Float64, N::Int64; decay="T", n=1, temp=1.0, pos=false)

    # Creates the 2-band objects in position space, by default
    # initial density matrix is built as a Gibbs distribution,
    # if flag pos is set to true the density matrix is initialized 
    # at a position eigenstate specified by n
    
    # Builds the Hamiltonian    
    H = hamiltonian_pos(t1, t2, V, N)

    # Initial density matrix
    
    if pos==false
        if temp != 0.0
            pops = exp.(-H.ens./(temp+0.0im))
            ρ0 = density_matrix(diagm(pops)./sum(pops))
        elseif temp==0.0
            pops = [i==1 ? (1.0+0.0im) : (0.0+0.0im) for i in 1:2*N]
            ρ0 = density_matrix(diagm(pops)./sum(pops))
        end
    elseif pos==true
        posdiag = [i==n ? 1.0+0.0im : 0.0+0.0im for i in 1:2*N]
        ρ0mat = diagm(posdiag)
        ρ0 = density_matrix(ρ0mat) 
    else
        error("pos can be only true or false")
    end

    # Builds the curent and the current
    # projector structs
    J = current_pos(t1, t2, N)
    
    if decay=="A"
        λ = build_decayA(γ, N)
    elseif decay=="B"
        λ = build_decayB(γ, N)
    elseif decay=="C"
        λ = build_decayC(γ, N)
    elseif decay=="T"
        λ = build_decayT(γ, H.ens, temp)
    else
        error("Decay can only be A, B, or C.")
    end
        
    x  = position_pos(N)
   
    return ρ0, H, x, J, λ
end

function hamiltonian_pos_coupled(t1::Float64, t2::Float64, V::Float64, A::Float64, N::Int64)
    
    H1 = zeros(ComplexF64, 2*N, 2*N)
    H2 = zeros(ComplexF64, 2*N, 2*N)
    H3 = zeros(ComplexF64, 2*N, 2*N)
    j1 = zeros(ComplexF64, 2*N, 2*N)
    j2 = zeros(ComplexF64, 2*N, 2*N)
    
    for i in 1:N
        H1[2*i-1, 2*i-1] = 1.0
        H1[2*i, 2*i] = -1.0
        
        H2[2*i-1, 2*i] = 1.0
        H2[2*i, 2*i-1] = 1.0
        
        j1[2*i-1, 2*i] = 1.0im
        j1[2*i, 2*i-1] = -1.0im
        
        i != N && (j2[2*i, 2*i+1] = 1.0im)
        i != N && (j2[2*i+1, 2*i] = -1.0im)
        
        i != N && (H3[2*i, 2*i+1] = 1.0)
        i != N && (H3[2*i+1, 2*i] = 1.0)
    end    
    
    H3[1, 2*N] = 1.0
    H3[2*N, 1] = 1.0
    
    j2[1, 2*N] = -1.0im
    j2[2*N, 1] = 1.0im  
    
    H = Hamiltonian([H1, H2, H3, j1, j2], (V/2, t1, t2, A*t1, A*t2), make)
    
    return H
end

function current_pos_coupled(t1::Float64, t2::Float64, A::Float64, N::Int64)
    
    j1 = zeros(ComplexF64, 2*N, 2*N)
    j2 = zeros(ComplexF64, 2*N, 2*N)
    H2 = zeros(ComplexF64, 2*N, 2*N)
    H3 = zeros(ComplexF64, 2*N, 2*N)
    
    for i in 1:N
        j1[2*i-1, 2*i] = 1.0im
        j1[2*i, 2*i-1] = -1.0im
        
        H2[2*i-1, 2*i] = 1.0
        H2[2*i, 2*i-1] = 1.0
        
        i != N && (H3[2*i, 2*i+1] = 1.0)
        i != N && (H3[2*i+1, 2*i] = 1.0)
        
        i != N && (j2[2*i, 2*i+1] = 1.0im)
        i != N && (j2[2*i+1, 2*i] = -1.0im)
    end    
    
    H3[1, 2*N] = 1.0
    H3[2*N, 1] = 1.0
    
    j2[1, 2*N] = -1.0im
    j2[2*N, 1] = 1.0im  
    
    j = observable([j1, j2, H2, H3], (t1, t2, A*t1, A*t2), make)
    
    return j
end

function build_objects_coupled(t1::Float64, t2::Float64, V::Float64, A::Float64,
                               γ::Float64, N::Int64; decay="T", n=1, temp=1.0, pos=false)

    # Creates the 2-band objects in position space, by default
    # initial density matrix is built as a Gibbs distribution,
    # if flag pos is set to true the density matrix is initialized 
    # at a position eigenstate specified by n
    
    # Builds the Hamiltonian    
    H = hamiltonian_pos_coupled(t1, t2, V, A, N)

    # Initial density matrix
    
    if pos==false
        temp == 0.0 && (temp=0.001)
        pops = exp.(-H.ens./(temp+0.0im))
        ρ0 = density_matrix(diagm(pops)./sum(pops))
    elseif pos==true
        posdiag = [i==n ? 1.0+0.0im : 0.0+0.0im for i in 1:2*N]
        ρ0mat = diagm(posdiag)
        ρ0 = density_matrix(ρ0mat) 
    else
        error("pos can be only true or false")
    end

    # Builds the curent and the current
    # projector structs
    J = current_pos_coupled(t1, t2, A, N)
    
    if decay=="A"
        λ = build_decayA(γ, N)
    elseif decay=="B"
        λ = build_decayB(γ, N)
    elseif decay=="C"
        λ = build_decayC(γ, N)
    elseif decay=="T"
        λ = build_decayT(γ, H.ens, temp)
    else
        error("Decay can only be A, B, or C.")
    end
        
    x  = position_pos(N)
   
    return ρ0, H, x, J, λ
end

function build_objects_2particle(t1::Float64, t2::Float64, V::Float64,
                                γ::Float64, N::Int64; temp=1.0)

    # Creates the 2-band objects in position space for 2 
    # spinless fermions
    
    # Builds the Hamiltonian    
    H = hamiltonian_pos(t1, t2, V, N)
    H2 = operator_single_fermi(H.matrix, operator=true)
    H2ham = Hamiltonian([H2.matrix], (1.0, ), make)
    
    
    # Initial density matrix
    temp == 0.0 && (temp=0.001)
    pops = exp.(-H2ham.ens./(temp+0.0im))
    ρ0 = density_matrix(diagm(pops)./sum(pops))

    # Builds the curent and the current
    # projector structs
    J = current_pos(t1, t2, N)
    J2 = operator_single_fermi(J.matrix, operator=true) 
    
    λ = build_decayT(γ, H2ham.ens, temp)
    
    x  = position_pos(N)
    x2 = operator_single_fermi(x.matrix, operator=true)
    
    return ρ0, H2ham, x2, J2, λ
end

function build_objects_2particle_int(t1::Float64, t2::Float64, V::Float64,
                                γ::Float64, Vs::Vector{ComplexF64}, N::Int64; temp=1.0)

    # Creates the 2-band objects in position space for 2 
    # spinless interacting fermions with interaction strengths
    # specified by Vs vector
    
    # Builds the Hamiltonian    
    H0 = hamiltonian_pos(t1, t2, V, N)
    H20 = operator_single_fermi(H0.matrix, operator=false)
    Vint = interaction_fermi(Vs, operator=false)
    H2ham = Hamiltonian([H20, Vint], (1.0, 1.0), make)
    
    
    # Initial density matrix
    temp == 0.0 && (temp=0.001)
    pops = exp.(-H2ham.ens./(temp+0.0im))
    ρ0 = density_matrix(diagm(pops)./sum(pops))

    # Builds the curent and the current
    # projector structs
    J = current_pos(t1, t2, N)
    J2 = operator_single_fermi(J.matrix, operator=true) 
    
    λ = build_decayT(γ, H2ham.ens, temp)
    
    x  = position_pos(N)
    x2 = operator_single_fermi(x.matrix, operator=true)
    
    return ρ0, H2ham, x2, J2, λ
end


#=====================================================================#
#--------------------------- Two Particles ---------------------------#
#=====================================================================#

function fermi_index_mapping(N::Int64, pnumber::Int64)
    # For a given Hilbert space dimension of single particle
    # and given particle number pnumber, the function returns
    # indices in the occupation number basis for fermions
    
    fermi_map = collect(combinations([1:1:N; ], pnumber))
    
    return fermi_map 
end

function operator_single_fermi(A1::Array{ComplexF64}; operator=true)
    # For two spinless fermions given a single particle operator
    # in a single-particle Hilbert space, it returns the operator
    # in a two-particle segment of the Fock space 
    
    N = size(A1)[1]
    index_map = fermi_index_mapping(N, 2)
    dim2 = N*(N-1)÷2
    
    A2 = Array{ComplexF64}(undef, dim2, dim2)
    
    for i in 1:dim2
        for j in 1:dim2
            entry = 0.0+0.0im
            bra_inds = index_map[i]
            ket_inds = index_map[j]
            if bra_inds[1]==ket_inds[1]
                entry+=A1[bra_inds[2], ket_inds[2]]
            end
            if bra_inds[2]==ket_inds[2]
                entry+=A1[bra_inds[1], ket_inds[1]]
            end
            if bra_inds[1]==ket_inds[2]
                entry-=A1[bra_inds[2], ket_inds[1]]
            end
            if bra_inds[2]==ket_inds[1]
                entry-=A1[bra_inds[1], ket_inds[2]]
            end
            A2[i, j] = entry
        end
    end
    
    if operator==true
        A2_operator = observable([A2], (1.0, ), make)
        return A2_operator
    elseif operator==false
        return A2
    end
end

function interaction_fermi(Vs::Vector{ComplexF64}; operator=true)
    # The interaction is specified by a vector
    # whose entries are interaction strengths
    # at each possible distance
    
    N = 2*length(Vs)
    index_map = fermi_index_mapping(N, 2)
    dim2 = N*(N-1)÷2
    
    V2 = zeros(ComplexF64, dim2, dim2)
    
    for i in 1:dim2
        entry = 0.0+0.0im
        inds = index_map[i]
        inddif_right = Int(inds[2]-inds[1])
        inddif_left = Int(inds[1]-inds[2]+N)
        inddif = min(inddif_right, inddif_left)
        V2[i, i] = Vs[inddif] 
    end
    
    if operator==true
        V2_operator = observable([V2], (1.0, ), make)
        return V2_operator
    elseif operator==false
        return V2
    end
end

#=====================================================================#
#------------------------- Three Band Model --------------------------#
#=====================================================================#

function build_position_three(N)
    
    ks = momenta_trunc(N, 3)
    xmat = zeros(ComplexF64, 3*N, 3*N)
    
    for i in 1:N
        xmat[3*i-2, 3*i-2] = (3*N-1)/4
        xmat[3*i-1, 3*i-1] = (3*N+1)/4
        xmat[3*i, 3*i] = (3*N+3)/4
    end

    for i in 1:N
        for j in i+1:N
            rij = exp(-3im*(ks.vector[i]-ks.vector[j]))
            Fij = (6*rij^2-3*N*rij*(1-rij))/(N*(1-rij)^2)
            xmat[3*i-2, 3*j-2] = Fij*exp(2im*(ks.vector[i]-ks.vector[j]))
            xmat[3*i-1, 3*j-1] = Fij*exp(1im*(ks.vector[i]-ks.vector[j]))
            xmat[3*i, 3*j] = Fij
        end
    end
    
    xmat = xmat+xmat'
    
    x = observable([xmat], (1.0, ), make)
    
    return x
end

function build_decaythreeA(γ::Float64, N::Int64)
    
    λmat = zeros(Float64, 3*N, 3*N)
    
    for i in 1:3*N-1
        λmat[i, i+1:end] = γ.*ones(3*N-i)
    end
    
    for i in 1:(3*N-1)÷2
        λmat[2*i+1, 2*i] = γ
    end
    
    if N%2 != 0
        for i in 1:N
            λmat[2*N+1, 2*N] = 0.0
            λmat[2*N, 2*N+2] = 0.0
            λmat[2*N, 2*N+3] = 0.0
        end
    elseif N%2 == 0
        for i in 1:N
            λmat[N+1, N] = 0.0
            λmat[N, N+2] = 0.0
            λmat[N, N+3] = 0.0  
            λmat[2*N+1, 2*N] = 0.0
            λmat[2*N, 2*N+2] = 0.0
            λmat[2*N, 2*N+3] = 0.0          
        end
    end
    
    λ = λmatrix(λmat)
    return λ
end

function build_decaythreeB(γ::Float64, N::Int64)
    
    λmat = zeros(Float64, 3*N, 3*N)
    
    for i in 1:3*N-1
        λmat[i, i+1] = γ
        i != 3*N-1 && (λmat[i, i+2] = γ)
    end
    
    for i in 2:2:3*N-1
       λmat[i+1, i] = γ 
    end
    
    for i in 2:2:3*N-3
        λmat[i, i+3] = γ
    end
    
    if N%2 != 0
        λmat[2*N+1, 2*N] = 0.0
        λmat[2*N-1, 2*N+1] = 0.0
        λmat[2*N-2, 2*N+1] = 0.0
        λmat[2*N, 2*N+2] = 0.0
        λmat[2*N, 2*N+3] = 0.0
        
    elseif N%2 == 0
        λmat[N+1, N] = 0.0
        λmat[N, N+2] = 0.0
        λmat[N, N+3] = 0.0  
        λmat[2*N+1, 2*N] = 0.0
        λmat[2*N, 2*N+2] = 0.0
        λmat[2*N, 2*N+3] = 0.0  
        λmat[N-1, N+1] = 0.0
        λmat[N-2, N+1] = 0.0
        λmat[2*N-1, 2*N+1] = 0.0
        λmat[2*N-2, 2*N+1] = 0.0
    end
    
    λ = λmatrix(λmat)
    return λ
end

function build_decaythreeC(γ::Float64, N::Int64)
    
    λmat = zeros(Float64, 3*N, 3*N)
    
    for i in 1:3*N-1
        λmat[i, i+1] = γ
        i != 3*N-1 && (λmat[i, i+2] = γ)
    end
    
    for i in 2:2:3*N-1
       λmat[i+1, i] = 0.0 
    end
    
    for i in 3:2:3*N
       λmat[i-1, i] = 0.0 
    end
    
    for i in 2:2:3*N-3
        λmat[i, i+3] = γ
    end
    
    if N%2 != 0
        λmat[2*N+1, 2*N] = 0.0
        λmat[2*N-1, 2*N+1] = 0.0
        λmat[2*N-2, 2*N+1] = 0.0
        λmat[2*N, 2*N+2] = 0.0
        λmat[2*N, 2*N+3] = 0.0
        λmat[2*N, 2*N+1] = γ
    elseif N%2 == 0
        λmat[N+1, N] = 0.0
        λmat[N, N+2] = 0.0
        λmat[N, N+3] = 0.0  
        λmat[2*N+1, 2*N] = 0.0
        λmat[2*N, 2*N+2] = 0.0
        λmat[2*N, 2*N+3] = 0.0  
        λmat[N-1, N+1] = 0.0
        λmat[N-2, N+1] = 0.0
        λmat[2*N-1, 2*N+1] = 0.0
        λmat[2*N-2, 2*N+1] = 0.0
        λmat[2*N, 2*N+1] = γ
        λmat[N, N+1] = γ
    end
    
    λ = λmatrix(λmat)
    return λ
end


function build_three(t1::Float64, t2::Float64, t3::Float64,
                     V1::Float64, V2::Float64, N::Int64)
    
    hargs = (t1, t2, t3, V1, V2)
    ks = momenta_trunc(N, 3)

    h1, h2, h3, h4, h5 = zeros(ComplexF64, 3*N, 3*N),
                         zeros(ComplexF64, 3*N, 3*N),
                         zeros(ComplexF64, 3*N, 3*N),
                         zeros(ComplexF64, 3*N, 3*N),
                         zeros(ComplexF64, 3*N, 3*N)
    
    for i in 1:N
        h1[3*i-2, 3*i-1] = exp(1im*ks.vector[i])
        h1[3*i-1, 3*i-2] = exp(-1im*ks.vector[i])
        h2[3*i-1, 3*i] = exp(1im*ks.vector[i])
        h2[3*i, 3*i-1] = exp(-1im*ks.vector[i])
        h3[3*i-2, 3*i] = exp(-1im*ks.vector[i])
        h3[3*i, 3*i-2] = exp(1im*ks.vector[i])
        h4[3*i-2, 3*i-2] = 1.0
        h5[3*i, 3*i] = 1.0
    end
    
    hs = [h1, h2, h3, h4, h5]
    
    return hs, hargs
end

function build_current_three(t1::Float64, t2::Float64,
                             t3::Float64, N::Int64)
    
    cargs = (t1, t2, t3)
    ks = momenta_trunc(N, 3)

    c1, c2, c3 = zeros(ComplexF64, 3*N, 3*N),
                 zeros(ComplexF64, 3*N, 3*N),
                 zeros(ComplexF64, 3*N, 3*N)
                     
    for i in 1:N
        c1[3*i-2, 3*i-1] = 1im*exp(1im*ks.vector[i])
        c1[3*i-1, 3*i-2] = -1im*exp(-1im*ks.vector[i])
        c2[3*i-1, 3*i] = 1im*exp(1im*ks.vector[i])
        c2[3*i, 3*i-1] = -1im*exp(-1im*ks.vector[i])
        c3[3*i-2, 3*i] = -1im*exp(-1im*ks.vector[i])
        c3[3*i, 3*i-2] = 1im*exp(1im*ks.vector[i])
    end
    
    cs = [c1, c2, c3]
    
    return cs, cargs
end

function build_objects_3band(t1::Float64, t2::Float64, t3::Float64,
                             V1::Float64, V2::Float64, γ::Float64,
                             N::Int64, n::Int64; decay="A", temp=1.0, en=false)

    ks = momenta_trunc(N, 3)
    if en==false
        ρ0 = initial_pos_three(ks, n)
    elseif en==true
        ρ0 = initial_en_three(ks, n)
    else
       error("en can be either true or false") 
    end
        
        
    # Builds the Hamiltonian    
    hs, hargs = build_three(t1, t2, t3, V1, V2, N)
    H = Hamiltonian(hs, hargs, make)

    # Builds the curent and the current
    # projector structs
    cs, cargs = build_current_three(t1, t2, t3, N)
    J = observable(cs, cargs, make)
    
    
    if decay=="A"
        λ = build_decaythreeA(γ, N)
    elseif decay=="B"
        λ = build_decaythreeB(γ, N)
    elseif decay=="C"
        λ = build_decaythreeC(γ, N)
    elseif decay=="T"
        λ = build_decayT(γ, H.ens, temp)
    else
        error("You can only choose models A, B, and C.")
    end

    x = build_position_three(N)
   
    return ρ0, H, x, J, λ
end

function hamiltonian_pos_three(t1::Float64, t2::Float64, t3::Float64, V1::Float64, V2::Float64, N::Int64)
    
    H1 = zeros(ComplexF64, 3*N, 3*N)
    H2 = zeros(ComplexF64, 3*N, 3*N)
    H3 = zeros(ComplexF64, 3*N, 3*N)
    H4 = zeros(ComplexF64, 3*N, 3*N)
    H5 = zeros(ComplexF64, 3*N, 3*N)
    
    for i in 1:N
        H1[3*i-2, 3*i-1] = 1.0
        H1[3*i-1, 3*i-2] = 1.0
        H2[3*i-1, 3*i] = 1.0
        H2[3*i, 3*i-1] = 1.0
        
        H4[3*i-2, 3*i-2] = 1.0
        H5[3*i, 3*i] = 1.0
        
        i != N && (H3[3*i, 3*i+1] = 1.0)
        i != N && (H3[3*i+1, 3*i] = 1.0)
    end    
    
    H3[1, 3*N] = 1.0
    H3[3*N, 1] = 1.0
    
    H = Hamiltonian([H1, H2, H3, H4, H5], (t1, t2, t3, V1, V2), make)
    
    return H
end

function current_pos_three(t1::Float64, t2::Float64, t3::Float64, N::Int64)
    
    j1 = zeros(ComplexF64, 3*N, 3*N)
    j2 = zeros(ComplexF64, 3*N, 3*N)
    j3 = zeros(ComplexF64, 3*N, 3*N)
    
    for i in 1:N
        j1[3*i-2, 3*i-1] = 1.0im
        j1[3*i-1, 3*i-2] = -1.0im
        j2[3*i-1, 3*i] = 1.0im
        j2[3*i, 3*i-1] = -1.0im
        
        i != N && (j3[3*i, 3*i+1] = 1.0im)
        i != N && (j3[3*i+1, 3*i] = -1.0im)
    end    
    
    j3[1, 3*N] = -1.0im
    j3[3*N, 1] = 1.0im  
    
    j = observable([j1, j2, j3], (t1, t2, t3), make)
    
    return j
end

function position_pos_three(N::Int64)
   
    xmat = zeros(ComplexF64, 3*N, 3*N)
    
    for i in 1:3*N
        xmat[i, i] = i
    end
    
    x = observable([xmat], (1.0, ), make)
    
    return x
end

function build_objects_3band_position(t1::Float64, t2::Float64, t3::Float64, V1::Float64, V2::Float64,
                                      γ::Float64, N::Int64; decay="T", n=1, temp=1.0, pos=false)

    # Creates the 2-band objects in position space, by default
    # initial density matrix is built as a Gibbs distribution,
    # if flag pos is set to true the density matrix is initialized 
    # at a position eigenstate specified by n
    
    # Builds the Hamiltonian    
    H = hamiltonian_pos_three(t1, t2, t3, V1, V2, N)

    # Initial density matrix
    
    if pos==false
        temp == 0.0 && (temp=0.001)
        pops = exp.(-H.ens./(temp+0.0im))
        ρ0 = density_matrix(diagm(pops)./sum(pops))
    elseif pos==true
        posdiag = [i==n ? 1.0+0.0im : 0.0+0.0im for i in 1:3*N]
        ρ0mat = diagm(posdiag)
        ρ0 = density_matrix(ρ0mat) 
    else
        error("pos can be only true or false")
    end

    # Builds the curent and the current
    # projector structs
    J = current_pos_three(t1, t2, t3, N)
    
    if decay=="A"
        λ = build_decaythreeA(γ, N)
    elseif decay=="B"
        λ = build_decaythreeB(γ, N)
    elseif decay=="C"
        λ = build_decaythreeC(γ, N)
    elseif decay=="T"
        λ = build_decayT(γ, H.ens, temp)
    else
        error("Decay can only be A, B, C, or T.")
    end
        
    x = position_pos_three(N)
   
    return ρ0, H, x, J, λ
end

function build_squares_3band(t1::Float64, t2::Float64,
                             t3::Float64, N::Int64)
    
    cs, cargs = build_current_three(t1, t2, t3, N)
    J = observable(cs, cargs, make)
    J2 = observable([J.matrix*J.matrix], (1.0, ), make)
    
    #x = build_position_three(N)
    #x2 = observable([x.matrix*x.matrix], (1.0, ), make)
    
    return J2
end

function build_occupation3(N; site="a")
   
    siteindex = Dict("a"=>2, "b"=>1, "c"=>0) 
    
    OccM = zeros(Float64, 3*N, 3*N)
    
    for i in 1:N
        OccM[3*i-siteindex[site], 3*i-siteindex[site]] = 1.0
    end
    
    Occ = observable([OccM], (1.0, ), make)
    
    return Occ
end




####################################################################################
########################### Projectors construction ################################
####################################################################################

function count_evals(cur::observable; dig=6)
    # Return the truncated eigenvalue list
    # and the other list that contains starting eigenvalues
    # with their truncated counts
     
    Pjs = Array{ComplexF64}(undef, cur.size,
                            cur.size, cur.size)
    
    curevals = [trunc(cureval, digits=dig) for cureval in cur.vals]
    curcount = Tuple{Float64, Int64}[]
    
    @assert cur.size>1 "You need to provide a larger matrix"
    
    i = 1
    while i<=cur.size
        eval_counter = 1
        j=i
        j == cur.size && (push!(curcount, (cur.vals[i], eval_counter)); break) 
        while curevals[j] == curevals[j+1]
            eval_counter += 1
            j<cur.size-1 ? j+=1 : break
        end
        push!(curcount, (cur.vals[i], eval_counter))
        i+=eval_counter
    end
    
    return curevals, curcount
end

struct Projectors
    size::Int64
    number::Int64
    shape::Tuple{Int64, Int64}
    matrices::Array{ComplexF64}
    en_basis::Array{ComplexF64}
    evals::Vector{Float64}
    counts::Vector{Int64}
    function Projectors(cur::observable, ham::Hamiltonian)
        
        curevals, curcount = count_evals(cur, dig=6)
        
        Pjs = Array{ComplexF64}(undef, cur.size, cur.size, length(curcount))
        Pjsϵ = Array{ComplexF64}(undef, cur.size, cur.size, length(curcount))
        
        # Starting counter
        # for the i-th projector
        j0 = 1
        
        for i in eachindex(curcount)
                count = curcount[i][2]
                Pjs[:, :, i] = sum([cur.vectors[:, j]*cur.vectors[:, j]'
                                    for j in j0:(j0+count-1)])
                j0+=count
        end
        
        for i in eachindex(curcount)
            Pjsϵ[:, :, i] = (ham.vectors')*Pjs[:, :, i]*ham.vectors
        end
            
        new(cur.size, length(curcount), cur.shape, Pjs, Pjsϵ,
            [curcount[i][1] for i in 1:length(curcount)], 
            [curcount[i][2] for i in 1:length(curcount)])
    end
end

function measurement_projectors(ϕ1::Float64, ϕ2::Float64, θ::Float64, N::Int64, U_H::Array{ComplexF64})

    # The function takes parameters of bond unitary and creates projectors for
    # complete measurement in the bond basis. The output of the function is a
    # 3D Matrix whose z-axis slices are 2N projectors that determine the measurement.
    # The output is given in the original (position) basis as well as in energy
    # basis.
    
    
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

struct ProjectorsTwo
    size::Int64
    number::Int64
    shape::Tuple{Int64, Int64}
    matrices::Array{ComplexF64}
    en_basis::Array{ComplexF64}
    function ProjectorsTwo(control::Array{ComplexF64}, ham::Hamiltonian, site_num::Int64)
        # The constructor takes a 2x2 bond control and creates 2*N
        # occupation projectors with binary choice for each site
    
    
        Pjs = Array{ComplexF64}(undef, ham.size, ham.size, 2*site_num)
        Pjsϵ = Array{ComplexF64}(undef, ham.size, ham.size, 2*site_num)
        
        for i in 1:site_num
            
            n1 = operator_single_fermi(diagm([j==2*i-1 ? 1.0+0im : 0.0+0.0im for j in 1:2*site_num])).matrix
            n2 = operator_single_fermi(diagm([j==2*i ? 1.0+0im : 0.0+0.0im for j in 1:2*site_num])).matrix
            
            n12help = zeros(ComplexF64, 2*site_num, 2*site_num)
            n21help = zeros(ComplexF64, 2*site_num, 2*site_num)
            
            n12help[2*i-1, 2*i] = 1.0+0.0im
            n21help[2*i, 2*i-1] = 1.0+0.0im
             
            # Create 1s at right places
            n12 = operator_single_fermi(n12help).matrix
            n21 = operator_single_fermi(n21help).matrix
            
            O1 = @. (abs(control'[1,1]))^2*n1+(abs(control'[1,2]))^2*n2+(control'[1,1])*conj(control'[1,2])*n12+(control'[1,2])*conj(control'[1,1])*n21
            O2 = @. (abs(control'[2,1]))^2*n1+(abs(control'[2,2]))^2*n2+(control'[2,2])*conj(control'[2,1])*n21+(control'[2,1])*conj(control'[2,2])*n12
            
            Pjs[:, :, 2*i-1] = O1 
            Pjs[:, :, 2*i] = O2
        end
        
        for i in 1:2*site_num
            Pjsϵ[:, :, i] = (ham.vectors')*Pjs[:, :, i]*ham.vectors
        end
            
        new(ham.size, 2*site_num, ham.shape, Pjs, Pjsϵ)
    end
end

struct ProjectorsOcc
    size::Int64
    number::Int64
    shape::Tuple{Int64, Int64}
    matrices::Array{ComplexF64}
    en_basis::Array{ComplexF64}
    function ProjectorsOcc(ham::Hamiltonian, control::Array{ComplexF64}, site_num::Int64)
        
        Pjs = Array{ComplexF64}(undef, ham.size, ham.size, 2)
        Pjsϵ = Array{ComplexF64}(undef, ham.size, ham.size, 2)
        
        bond = zeros(ComplexF64, 2*site_num, 2*site_num)
        bond[1:2, 1:2] = control*[1.0+0.0im 0.0; 0.0 0.0]*control'
        
        n1 = operator_single_fermi(bond).matrix
        not1 = diagm(ones(ComplexF64, ham.size))-n1
        
        Pjs[:, :, 1] = n1
        Pjs[:, :, 2] = not1
        
        for i in 1:2
            Pjsϵ[:, :, i] = (ham.vectors')*Pjs[:, :, i]*ham.vectors
        end
        new(ham.size, 2, ham.shape, Pjs, Pjsϵ)
    end
end

struct KrausOperators
    size::Int64
    number::Int64
    shape::Tuple{Int64, Int64}
    matrices::Array{ComplexF64}
    en_basis::Array{ComplexF64}
    function KrausOperators(control::Array{ComplexF64}, ham::Hamiltonian, site_num::Int64, ξ::Float64)
        # The constructor takes a 2x2 bond control and creates 2*N
        # occupation projectors with binary choice for each site
    
    
        Ms = Array{ComplexF64}(undef, ham.size, ham.size, 2*site_num+1)
        Msϵ = Array{ComplexF64}(undef, ham.size, ham.size, 2*site_num+1)
        
        one = [1.0 0.0; 0.0 0.0]
        two = [0.0 0.0; 0.0 1.0]
        
        for i in 1:site_num
            
            M0 = diagm(zeros(ComplexF64, 2*site_num))
            M1 = diagm(zeros(ComplexF64, 2*site_num))
            
            M0[2*i-1:2*i, 2*i-1:2*i].= cos(ξ)*control*one*control'+sin(ξ)*control*two*control'
            M1[2*i-1:2*i, 2*i-1:2*i].= sin(ξ)*control*one*control'+cos(ξ)*control*two*control'
            
            Ms[:, :, 2*i-1] = M0 
            Ms[:, :, 2*i] = M1
        end
        
        Ms[:, :, 2*site_num+1] = sqrt(diagm(ones(ComplexF64, 2*site_num)).-sum(Ms[:, :, 1:2*site_num]))
        
        for i in 1:2*site_num+1
            Msϵ[:, :, i] = (ham.vectors')*Ms[:, :, i]*ham.vectors
        end
            
        new(ham.size, 2*site_num+1, ham.shape, Ms, Msϵ)
    end
end

struct Kraus
    size::Int64
    number::Int64
    shape::Tuple{Int64, Int64}
    matrices::Array{ComplexF64}
    en_basis::Array{ComplexF64}
    function Kraus(Pjs::Projectors, ham::Hamiltonian, control::Array{ComplexF64}, ϕ::Float64, b::Float64, Ψ::Float64; single_site=false)
        # The constructor takes projector matrices
        # and returns the Kraus operators
        
        single_site==false ? (number=Pjs.number) : (number=2)
    
        Ms = Array{ComplexF64}(undef, ham.size, ham.size, number)
        Msϵ = Array{ComplexF64}(undef, ham.size, ham.size, number)
        
        for i in 1:Int(Pjs.number ÷2)
            Pjs.matrices[2*i-1:2*i, 2*i-1:2*i, 2*i-1] = control*Pjs.matrices[2*i-1:2*i, 2*i-1:2*i, 2*i-1]*control'
            Pjs.matrices[2*i-1:2*i, 2*i-1:2*i, 2*i] = control*Pjs.matrices[2*i-1:2*i, 2*i-1:2*i, 2*i]*control'
        end
        
        if single_site==false
            for i in 2:number-1
                Ms[:, :, i] = cos(ϕ)*Pjs.matrices[:, :, i]+exp(-1im*Ψ)*sqrt(0.5-b/2)*sin(ϕ)*Pjs.matrices[:, :, i-1]+exp(1im*Ψ)*sqrt(0.5+b/2)*sin(ϕ)*Pjs.matrices[:, :, i+1]
            end
            Ms[:, :, 1] = cos(ϕ)*Pjs.matrices[:, :, 1]+exp(-1im*Ψ)*sqrt(0.5-b/2)*sin(ϕ)*Pjs.matrices[:, :, number]+exp(1im*Ψ)*sqrt(0.5+b/2)*sin(ϕ)*Pjs.matrices[:, :, 2]
            Ms[:, :, number] = cos(ϕ)*Pjs.matrices[:, :, number]+exp(-1im*Ψ)*sqrt(0.5-b/2)*sin(ϕ)*Pjs.matrices[:, :, number-1]+exp(1im*Ψ)*sqrt(0.5+b/2)*sin(ϕ)*Pjs.matrices[:, :, 1]
        elseif single_site==true
            Ms[:, :, 1] = cos(ϕ)*Pjs.matrices[:, :, 1]+exp(-1im*Ψ)*sqrt(0.5-b/2)*sin(ϕ)*Pjs.matrices[:, :, number]+exp(1im*Ψ)*sqrt(0.5+b/2)*sin(ϕ)*Pjs.matrices[:, :, 2]
            Ms[:, :, 2] = sqrt(diagm(ones(ComplexF64, ham.size)).-Ms[:, :, 1]'*Ms[:, :, 1])
        end
        
        for i in 1:number
            Msϵ[:, :, i] = (ham.vectors')*Ms[:, :, i]*ham.vectors
        end
            
        new(ham.size, number, ham.shape, Ms, Msϵ)
    end
end


##############################################################################
########################## Lindbladian construction ##########################
##############################################################################

struct λmatrix
    size::Int64
    shape::Tuple{Int64, Int64}
    matrix::Array{Float64}
    function λmatrix(λm::Array{Float64})
        new(size(λm)[1], (size(λm)[1], size(λm)[1]), λm) 
    end
end
    
struct λsum
    size::Int64
    vector::Vector{Float64}
    function λsum(λ::λmatrix)
        new(λ.size, [sum(λ.matrix[:, i]) for i in 1:λ.size])
    end
end

struct MMatrix
    size::Int64
    shape::Tuple{Int64, Int64}
    matrix::Array{Float64}
    function MMatrix(Mm::Array{Float64})
        for i in size(Mm)[1]
            @assert isapprox(sum(Mm[:, i]), 0.0, atol=1e-7) "The Mmatrix columns don't sum up to 0"
        end
        new(size(Mm)[1], (size(Mm)[1], size(Mm)[1]), Mm) 
    end
end

function λtoM(λ::λmatrix)::MMatrix
    
    Mm = copy(λ.matrix)
    Mm[diagind(Mm)] = zeros(Float64, λ.size)
    
    for i in 1:λ.size
        Mm[i, i] = -sum(Mm[:, i])
    end
    
    M = MMatrix(Mm)
    return M
end

function complex_freq(i::Int64, j::Int64, energy::Vector{Float64}, λsum::λsum)  
    # Takes energies and λmatrix
    # and returns complex frequencies
    
    Ωij = energy[i]-energy[j]-0.5im*(λsum.vector[i]+λsum.vector[j])
    return Ωij
end

function diagonalize(λmatrix::λmatrix)
    # Diagonalization method 
    λdiag, θ = eigen(λmatrix.matrix)
    return λdiag, θ
end

struct freqarray
    # Create an array of complex frequencies Ωij
    
    size::Int64
    shape::Tuple{Int64, Int64}
    matrix::Array{ComplexF64}
    function freqarray(Ens::Vector{Float64}, λsum::λsum)
        Ωarray = Array{ComplexF64}(undef, λsum.size, λsum.size)
        for i in 1:λsum.size
            for j in 1:λsum.size
                Ωarray[i, j] = complex_freq(i, j, Ens, λsum)
            end
        end
        new(λsum.size, (λsum.size, λsum.size), Ωarray)
    end
end

struct freqarray_exp
    # Creates an array of complex frequency exponentials
    # for time t --> exp(-iΩij*t)
    
    size::Int64
    shape::Tuple{Int64, Int64}
    time::Float64
    matrix::Array{ComplexF64}
    function freqarray_exp(f::freqarray, t::Float64)
        # The constructor contains element-wise exponentiation
        # of the frequency matrix and not the matrix exponentiation
        
        mexp = exp.(-1im*f.matrix*t)
        mexp[diagind(mexp)] = ones(f.size)
        new(f.size, f.shape, t, mexp)
    end
end


###################################################################################
########################### Population evolution ##################################
###################################################################################
    
function pop_evolve!(p::Vector{Float64}, expM::Array{Float64})
    p.=expM*p
    return p
end

function populations_exp!(p0::Vector{Float64}, M::MMatrix, dt::Float64, T::Float64,  N::Int)
    # Evolves the populations using the matrix exponentiation algorithm 
    
    ts = [0.0:dt:T; ]
    
    pops = Array{Float64}(undef, N, length(ts))
    pops[:, 1] = p0
    
    p = p0
    
    for i in 2:length(ts)
       pops[:, i] = pop_evolve!(p, exp(M.matrix*dt))
    end
    
    return ts, pops
end
    
function popevo_animate(pops::Array{Float64}, step::Int64; save=false,
                        frames=15, name="test_name.gif")
    # The function to animate the evoltion of the population
    # that takes as an argument an output of populations!-algorithms
    
    anim = @animate for i in 1:step:(size(pops)[2])
        bar([1:1:(size(pops)[1]);], pops[:, i])
    end
    
    save==true && (gif(anim, name, fps=frames))
    
    return anim
end

#---------------- Adjacent Flat Model of Decay ------------------#
#----------------       Analytic Solution      ------------------#
    
function λadjacent(N::Int64, γ::Float64)
    λm = zeros(Float64, N, N)
    for i in 1:N
       i != N && (λm[i, i+1]=γ) 
    end
    λ = λmatrix(λm)
end

function Madjacent(N::Int64, γ::Float64)
    Mm = zeros(Float64, N, N)
    for i in 1:N
        i !=1 && (Mm[i, i] = -γ) 
        i !=N && (Mm[i, i+1] = γ)
    end
    M = MMatrix(Mm)  # Creates a λmatrix from the given array
    return M
end

function I(a, x)
        # The incomplete gamma function that is part of the solution 
        # for the density matrix evolution of the adjacent flat model
        
    return a !=0 ? gamma(a, 0.0)*gamma_inc(a, x)[1] : x>=0.0 && 1.0
end

function expM_adjacent(γ::Float64, dt::Float64, N::Int64)
    # Population evolution matrix for the adjacent flat model
    # The results follow from the analytic solution of the 
    # differential equation of the problem

    expM = zeros(Float64, N, N)
    
    expM[1, :] = [i !=1 ? gamma_inc(i-1, γ*dt)[1] : 1.0 for i in 1:N]
    
    for i in 2:N
        for j in 0:N-i
            expM[i, i+j] = (exp(-γ*dt)*(γ*dt)^j)/gamma(j+1)
        end
    end
        
    return expM
end

function pop_evolve_adjacent(p::Vector{Float64}, γ::Float64, t::Float64)
    #Evolves the populations according to the adjacent flat lambda matrix
    
    expM = expM_adjacent(γ, t, length(p))

    return expM*p
end


function pop_evolve_adjacent!(p::Vector{Float64}, expM::Array{Float64})
    # Inplace version of the method
    
    p.=expM*p

    return p
end

function populations_adjacent!(p0::Vector{Float64}, dt::Float64, T::Float64, γ::Float64, N::Int)
    # Evolves the populations and records each state population for a given
    # time and sampling rate --> adjacent decay model algorithm
    
    expM = expM_adjacent(γ, dt, N)
    ts = [0.0:dt:T; ]
    
    pops = Array{Float64}(undef, N, length(ts))
    pops[:, 1] = p0
    
    
    for i in 2:length(ts)
       pops[:, i] = pop_evolve_adjacent!(pops[:, i-1], expM)
    end
    
    return ts, pops
end

###############################################################################
########################### Measurement methods ###############################
###############################################################################


function get_probs(ρ::Array{ComplexF64}, Pjs::Projectors)
    
    probs = Vector{Float64}(undef, Pjs.number)
    
    for i in 1:Pjs.number
        p = tr(Pjs.en_basis[:, :, i]*ρ)
        @assert isapprox(abs(imag(p)), 0.0, atol=1e-7) "The probabilite are not real p=$(p)"
        probs[i] = real(p)
    end
    
    return probs
end

function choose(r, probs)
    
    @assert isapprox(sum(probs), 1.0, atol=1e-7) "The probabilites don't sum up to one\n sum=$(sum(probs))"
    cum = 0.0
    choice = nothing
    
    for i in eachindex(probs)
        cum+=probs[i]
        if r<cum
            choice=i
            return choice
        end
    end
    return error("Didn't get to choose!")
end

function measure_and_pick!(ρ::Array{ComplexF64}, Op::Array{ComplexF64},
                          Pjs::Projectors, measured::Vector{Float64})
    # The function takes the density matrix and observable projectors as arguments, 
    # calculates the probabilities of different observable measurements, and returns
    # a single density matrix projection as an argument 
    
    probs = get_probs(ρ, Pjs)
    
    r = rand()
    choice = choose(r, probs)

    ρ .= Pjs.en_basis[:, :, choice]*ρ*Pjs.en_basis[:, :, choice]
    ρ .= ρ/tr(ρ)
    #push!(measured, Pjs.evals[choice])
    push!(measured, real(tr(Op*ρ)))
    
    # ADD THIS ASSERT STATEMENT!
    @assert isapprox(Pjs.evals[choice], tr(ρ*Op), atol=1e-7) "Values don't match"
    
end

function measure_and_pick_choices!(ρ::Array{ComplexF64}, Pjs::Projectors, choices::Vector{Int64})
    # The function takes the density matrix and observable projectors as arguments, 
    # calculates the probabilities of different observable measurements, and returns
    # a single density matrix projection as an argument 
    
    probs = get_probs(ρ, Pjs)
    
    r = rand()
    choice = choose(r, probs)
    push!(choices, choice)

    ρ .= Pjs.en_basis[:, :, choice]*ρ*Pjs.en_basis[:, :, choice]
    ρ .= ρ/tr(ρ)

    return ρ
end


#################################################################################
########################### ALGORITHMS AND INTERFACE ############################
###########################       FUNCTIONS          ############################
#################################################################################

#=====================================================================#
#------------------------   Single Run      --------------------------#
#------------------------    Algorithm      --------------------------#
#=====================================================================#

function current_adjacent(N::Int64, n::Int64, dt::Float64, T::Float64,
                          t1::Float64, t2::Float64, V::Float64, γ::Float64)
    
    # Start from the initial energy eigenstate with energy n and for a time T
    # evolve with the time-step dt the adjacent flat decay ratchet model with
    # arguments t1, t2, and V. Return the expectation value of the current at 
    # the measured time-steps
      
    ts = [0.0:dt:T; ]
    ks = momenta(N)
    ρ0 = dm_initial_pos(ks, n)

    test_dm(ρ0.matrix, print=false)
    
    hs, args = ratchet_build(t1, t2, V, N)
    H = Hamiltonian(hs, hargs, make)
    λm = λadjacent(2*N, γ) 
    γs = λsum(λm)
    
    freq = freqarray(H.ens, γs)
    expfreq = freqarray_exp(freq, dt)
    expM = expM_adjacent(γ, dt, 2*N)
    
    measured = []

    #This part should be customized:
    ks = momenta(N)
    tjs = current_integral(t1, t2, ks)
    jmom = current(tjs)
    j = H.vectors'*jmom.matrix*H.vectors 
    ρ = H.vectors'*ρ0.matrix*H.vectors
    push!(measured, real(tr(j*ρ)))

    
    for i in 2:length(ts)
        evolve_adjacent!(ρ, expfreq, expM, j, measured)
    end
    
    @assert isapprox(tr(ρ), 1.0, atol=1e-7) "At the end of the evolution the density
                                            matrix trace should still be unchanged" 
    
    return ts, measured
end
    
function current_measure_adjacent(N::Int64, n::Int64, dt::Float64, T::Float64, ωm::Float64,
                                  t1::Float64, t2::Float64, V::Float64, γ::Float64)
    
    # Start from the initial energy eigenstate with energy n and for a time T
    # evolve with the time-step dt the adjacent flat decay ratchet model with
    # arguments t1, t2, and V. Return the expectation value of the current at 
    # the measured time-steps
      
    ts = [0.0:dt:T; ]
    ks = momenta(N)
    ρ0 = dm_initial_pos(ks, n)
    test_dm(ρ0.matrix, print=false)

    
    # Builds the Hamiltonian    
    hs, hargs = ratchet_build(t1, t2, V, N)
    H = Hamiltonian(hs, hargs, make)
        
    # Builds the curent and the current
    # projector structs
    cs, cargs = ratchet_build_current(t1, t2, N)
    cur = observable(cs, cargs, make)
    jmom = cur.matrix
    Pjs = Projectors(cur, H)
    test_projectors(Pjs.matrices, print=false)
    
    ωm != 0.0 ? (nm = (1÷(dt*ωm))) : (nm=typemax(Int))
    
    # Builds the decay structs
    λm = λadjacent(2*N, γ) 
    γs = λsum(λm)
        
    # Builds the Lindbladian structs
    freq = freqarray(H.ens, γs)
    expfreq = freqarray_exp(freq, dt)
    expM = expM_adjacent(γ, dt, 2*N)
    
    measured = []

    # The observable that is measured;
    # This part should be customized:
    jϵ = H.vectors'*jmom*H.vectors 
    ρ = H.vectors'*ρ0.matrix*H.vectors
    push!(measured, real(tr(jϵ*ρ)))
    
    for i in 2:length(ts)
        if i%nm != 0
            evolve_adjacent!(ρ, expfreq, expM, jϵ, measured)
        else
            ρ = measure_and_pick!(ρ, Pjs, measured)
        end
    end
    
    @assert isapprox(tr(ρ), 1.0, atol=1e-7) "At the end of the evolution the density
                                            matrix trace should still be unchanged" 
    
    return ts, measured
end

#=====================================================================#
#------------------------  Adjacent Flat    --------------------------#
#------------------------    Algorithm      --------------------------#
#=====================================================================#


function evolve_adjacent!(ρ::Array{ComplexF64}, freqexp::freqarray_exp, expM::Array{Float64}, 
                          Op::Array{ComplexF64}, measured::Vector)
    
    # Starting from the initial density matrix ρ evolve the density
    # matrix for the time Δt and push the expectation value of the
    # measured observable in the "measured" container
    # THIS CAN BE CHANGED TO ADD THE SAMPLING FREQUENCY OF THE OBSERVABLE
    
    ρ.=freqexp.matrix.*ρ
    ρ[diagind(ρ)]=expM*(ρ[diagind(ρ)])

    push!(measured, real(tr(Op*ρ)))

    return ρ # <----The evolved density matrix
end

function evolve_adjacent_choices!(ρ::Array{ComplexF64}, freqexp::freqarray_exp, expM::Array{Float64})
    
    # Starting from the initial density matrix ρ evolve the density
    # matrix for the time Δt and push the expectation value of the
    # measured observable in the "measured" container
    # THIS CAN BE CHANGED TO ADD THE SAMPLING FREQUENCY OF THE OBSERVABLE
    
    ρ.= freqexp.matrix.*ρ
    ρ[diagind(ρ)]=expM*(ρ[diagind(ρ)])
   
    return ρ
end

function _evolve_measure!(ρ::Array{ComplexF64}, ts::Vector{Float64}, expfreq::freqarray_exp,
                          expM::Array{Float64}, jϵ::Array{ComplexF64}, nm::Int64,
                          Pjs::Projectors, measured::Vector{Float64})
    # Helper functions that executes one history
    
    push!(measured, real(tr(jϵ*ρ)))
    
    for i in 2:length(ts)
        if i%nm != 0
            evolve_adjacent!(ρ, expfreq, expM, jϵ, measured)
        else
            measure_and_pick!(ρ, jϵ, Pjs, measured)
        end
    end
    
    return measured
end

function current_ensemble_adjacent(N::Int64, n::Int64, nE::Int64, dt::Float64, T::Float64,
                                   ωm::Float64, t1::Float64, t2::Float64, V::Float64, γ::Float64)
     
    # Start from the initial energy eigenstate with energy n and for a time T
    # evolve with the time-step dt the adjacent flat decay ratchet model with
    # arguments t1, t2, and V. Return the expectation value of the current at 
    # the measured time-steps
      
    ts = [0.0:dt:T; ]
    ks = momenta(N)
    ρ0 = dm_initial(2*N, n)
    test_dm(ρ0.matrix, print=false)
    
    # Builds the Hamiltonian    
    hs, hargs = ratchet_build(t1, t2, V, N)
    H = Hamiltonian(hs, hargs, make)

    # Builds the curent and the current
    # projector structs
    cs, cargs = ratchet_build_current(t1, t2, N)
    cur = observable(cs, cargs, make)
    jmom = cur.matrix
    Pjs = Projectors(cur, H)
    test_projectors(Pjs.matrices, print=false)
    
    ωm != 0.0 ? (nm = Int(1÷(dt*ωm))) : (nm=typemax(Int))

    # Builds the decay structs
    λm = λadjacent(2*N, γ) 
    γs = λsum(λm)
        
    # Builds the Lindbladian structs
    freq = freqarray(H.ens, γs)
    expfreq = freqarray_exp(freq, dt)
    expM = expM_adjacent(γ, dt, 2*N)
    
    # The observable that is measured;
    # This part should be customized:
    jϵ = H.vectors'*jmom*H.vectors 
    ρ = H.vectors'*ρ0.matrix*H.vectors

    ρ0 = dm_initial(2*N, n)
    ρ = ρ0.matrix  

    cur_ensemble = zeros(Float64, length(ts))
    
    for i in 1:nE
        ρe = copy(ρ)
        meas = Float64[]
        cur_ensemble .+= _evolve_measure!(ρe, ts, expfreq, expM, jϵ, nm, Pjs, meas)./nE
    end
    
    return ts, cur_ensemble
end

#=====================================================================#
#------------------------ Choice Counting   --------------------------#
#------------------------    Algorithm      --------------------------#
#=====================================================================#

function _evolve_measure_choice!(ρ::Array{ComplexF64}, ts::Vector{Float64}, expfreq::freqarray_exp,
                          expM::Array{Float64}, nm::Int64,
                          Pjs::Projectors, choices)
    # Helper functions that executes one history
    
    for i in 2:length(ts)
        if i%nm != 0
            evolve_adjacent_choices!(ρ, expfreq, expM)
        else
            ρ = measure_and_pick_choices!(ρ, Pjs, choices)
        
        end
    end
    
    return choices
end

function count_choices(ccount, choices)
    
    # The function that takes the output of the choices
    # interface function and retuns counts of each choice
    # normalized version and current that is calculated
    # from the probabilities
    
    num = length(ccount)
    cc = Tuple{Float64, Int64}[]
    
    for i in 1:num
       push!(cc, (ccount[i][1], count(j->j==i, choices)))
    end
    
    numberofc = length(choices)
    probs = [cc[i][2]/numberofc for i in 1:length(cc)]
    current = sum(probs.*[cc[i][1] for i in 1:length(cc)])
    
    return cc, probs, current
end

function current_ensemble_choices(N::Int64, n::Int64, nE::Int64, dt::Float64, T::Float64,
                                   ωm::Float64, t1::Float64, t2::Float64, V::Float64, γ::Float64)
    
    # Start from the initial energy eigenstate with energy n and for a time T
    # evolve with the time-step dt the adjacent flat decay ratchet model with
    # arguments t1, t2, and V. Return the expectation value of the current at 
    # the measured time-steps
      
    ts = [0.0:dt:T; ]
    ks = momenta(N)
    ρ0 = dm_initial(2*N, n)
    test_dm(ρ0.matrix, print=false)
    
    # Builds the Hamiltonian    
    hs, hargs = ratchet_build(t1, t2, V, N)
    H = Hamiltonian(hs, hargs, make)
        
    # Builds the curent and the current
    # projector structs
    cs, cargs = ratchet_build_current(t1, t2, N)
    cur = observable(cs, cargs, make)
    cur_evals, cur_count = count_evals(cur)
    jmom = cur.matrix
    Pjs = Projectors(cur, H)
    test_projectors(Pjs.matrices, print=false)
    
    ωm != 0.0 ? (nm = Int(1÷(dt*ωm))) : (nm=typemax(Int))
    
    # Builds the decay structs
    λm = λadjacent(2*N, γ) 
    γs = λsum(λm)
        
    # Builds the Lindbladian structs
    freq = freqarray(H.ens, γs)
    expfreq = freqarray_exp(freq, dt)
    expM = expM_adjacent(γ, dt, 2*N)

    ρ = ρ0.matrix
    choices = Int64[]
    
    jϵ = H.vectors'*jmom*H.vectors 
    
    for i in 1:nE
        ρ0 = copy(ρ)
        _evolve_measure_choice!(ρ0, ts, expfreq, expM, nm, Pjs, choices)/nE
    end
    
    cc, probs, curav = count_choices(cur_count, choices)
    
    return choices, [cc, probs, curav]
end

#=====================================================================#
#------------------------ General Algorithm --------------------------#
#------------------------    Stochastic     --------------------------#
#=====================================================================#

function _measure!(ρ::Array{ComplexF64}, Op::Array{ComplexF64}, Pjs::Projectors, measured::Vector{Float64})
    # The function takes the density matrix and observable projectors as arguments, 
    # calculates the probabilities of different current measurements, and returns
    # a single density matrix projection as an argument 
    
    probs = get_probs(ρ, Pjs)
    
    r = rand()
    choice = choose(r, probs)

    ρ .= Pjs.en_basis[:, :, choice]*ρ*Pjs.en_basis[:, :, choice]
    ρ .= ρ/tr(ρ)
    
    push!(measured, real(tr(Op*ρ)))
end

function _evolve_decay!(ρ::Array{ComplexF64}, freqexp::freqarray_exp, expM::Array{Float64}, 
                          Op::Array{ComplexF64}, measured::Vector{Float64})
    
    # Starting from the initial density matrix ρ evolve the density
    # matrix for the time Δt and push the expectation value of the
    # measured observable in the "measured" container
    # THIS CAN BE CHANGED TO ADD THE SAMPLING FREQUENCY OF THE OBSERVABLE
    
    ρ.=freqexp.matrix.*ρ
    ρ[diagind(ρ)]=expM*(ρ[diagind(ρ)])    
    push!(measured, real(tr(Op*ρ)))

end

function _evolve!(ρ::Array{ComplexF64}, ts::Vector{Float64}, expfreq::freqarray_exp,
                          expM::Array{Float64}, jϵ::Array{ComplexF64}, nm::Int64,
                          Pjs::Projectors, measured::Vector{Float64})
    # Helper functions that executes one history
    
    push!(measured, real(tr(jϵ*ρ)))
    
    for i in 2:length(ts)
        if i%nm != 0
            _evolve_decay!(ρ, expfreq, expM, jϵ, measured)

        else
            _measure!(ρ, jϵ, Pjs, measured)
        end
    end
    
    return measured
end

function current_ensemble(ρ0::density_matrix, nE::Int64, 
                          dt::Float64, T::Float64, ωm::Float64,
                          H::Hamiltonian, cur::observable, λ::λmatrix)
    
    # Start from the initial energy eigenstate with energy n and for a time T
    # evolve with the time-step dt the adjacent flat decay ratchet model with
    # arguments t1, t2, and V. Return the expectation value of the current at 
    # the measured time-steps
    
    ts = [0.0:dt:T; ]
    
        
    # Builds the curent and the current
    # projector struct
    jmom = cur.matrix
    Pjs = Projectors(cur, H)
    #test_projectors(Pjs)
    ωm != 0.0 ? (nm = Int(1÷(dt*ωm))) : (nm=typemax(Int))

    # Builds the decay structs
    γs = λsum(λ)
        
    # Builds the Lindbladian structs
    M = λtoM(λ)
    freq = freqarray(H.ens, γs)
    expfreq = freqarray_exp(freq, dt)
    expM = exp(M.matrix*dt)

    jϵ = H.vectors'*jmom*H.vectors 
    ρ = ρ0.matrix

    cur_ensemble = zeros(Float64, length(ts))
    
    for i in 1:nE
        ρe = copy(ρ)
        meas = Float64[]
        cur_ensemble .+= _evolve!(ρe, ts, expfreq, expM, jϵ, nm, Pjs, meas)./nE
    end
    
    
    
    return ts, cur_ensemble
end

#=====================================================================#
#------------------------ General Algorithm --------------------------#
#=====================================================================#


function _measure_ratchet!(ρ::Array{ComplexF64}, Op::Array{ComplexF64}, Pjs::Projectors, measured::Vector{Float64})
    # The function takes the density matrix and observable projectors as arguments, 
    # calculates the probabilities of different current measurements, and returns
    # a single density matrix projection as an argument 
    
    
    ρbefore = copy(ρ)
    ρ.= zeros(ComplexF64, size(ρbefore)[1], size(ρbefore)[2])
    
    for i in 1:Pjs.number
        ρ .+= Pjs.en_basis[:, :, i]*ρbefore*Pjs.en_basis[:, :, i]
    end
    
    push!(measured, real(tr(Op*ρ)))
end


function _evolve_ratchet!(ρ::Array{ComplexF64}, ts::Vector{Float64}, expfreq::freqarray_exp,
                          expM::Array{Float64}, jϵ::Array{ComplexF64}, nm::Int64,
                          Pjs::Projectors, measured::Vector{Float64})
    # Helper function that executes one history
    
    push!(measured, real(tr(jϵ*ρ)))
    
    for i in 2:length(ts)
        #show(stdout, "text/plain", round.(ρ, digits=2))
        if i%nm != 0
            _evolve_decay!(ρ, expfreq, expM, jϵ, measured)
        else
            _measure_ratchet!(ρ, jϵ, Pjs, measured)
        end
    end
    
    return measured
end


function ratchet_measure!(ρ0::density_matrix, dt::Float64,
                          T::Float64, ωm::Float64,
                          H::Hamiltonian, x::observable, cur::observable, λ::λmatrix;
                          transform=false)
    
    # Start from the initial energy eigenstate with energy n and for a time T
    # evolve with the time-step dt the adjacent flat decay ratchet model with
    # arguments t1, t2, and V. Return the expectation value of the current at 
    # the measured time-steps

    # By default the density matrix is given in energy eigenbasis,
    # to change this use transform flag
    
    ts = [0.0:dt:T; ]
    
        
    # Builds the current and the observable
    # projector struct
    jmom = cur.matrix
    
    Pjs = Projectors(x, H)
    test_projectors(Pjs, print=false)

    test_dm(ρ0.matrix, print=false)
    
    ωm != 0.0 ? (nm = Int(1÷(dt*ωm))) : (nm=typemax(Int))
    
    # Builds the decay structs
    γs = λsum(λ)
        
    # Builds the Lindbladian structs
    M = λtoM(λ)
    freq = freqarray(H.ens, γs)
    expfreq = freqarray_exp(freq, dt)
    expM = exp(M.matrix*dt)

    jϵ = H.vectors'*jmom*H.vectors 
    transform == true ? (ρ = H.vectors'*ρ0.matrix*H.vectors) : (ρ = ρ0.matrix)
    currents = zeros(Float64, length(ts))
    
    meas = Float64[]
    currents = _evolve_ratchet!(ρ, ts, expfreq, expM, jϵ, nm, Pjs, meas)
    
    test_dm(ρ, print=false)
    #show(stdout, "text/plain", round.(ρ, digits=2))
    
    return ts, currents
end

function ratchet_measure_step!(ρ0::density_matrix, dt::Float64,
                          T::Float64, nm::Int64,
                          H::Hamiltonian, x::observable, cur::observable, λ::λmatrix;
                          transform=false)
    
    # Start from the initial energy eigenstate with energy n and for a time T
    # evolve with the time-step dt the adjacent flat decay ratchet model with
    # arguments t1, t2, and V. Return the expectation value of the current at 
    # the measured time-steps

    # By default the density matrix is given in energy eigenbasis,
    # to change this use transform flag
    
    ts = [0.0:dt:T; ]
    
        
    # Builds the current and the observable
    # projector struct
    jmom = cur.matrix
    
    Pjs = Projectors(x, H)
    test_projectors(Pjs, print=false)

    test_dm(ρ0.matrix, print=false)
    
    # Builds the decay structs
    γs = λsum(λ)
        
    # Builds the Lindbladian structs
    M = λtoM(λ)
    freq = freqarray(H.ens, γs)
    expfreq = freqarray_exp(freq, dt)
    expM = exp(M.matrix*dt)
    #expM = LinearAlgebra.I+M.matrix*dt

    jϵ = H.vectors'*jmom*H.vectors 
    transform == true ? (ρ = H.vectors'*ρ0.matrix*H.vectors) : (ρ = ρ0.matrix)
    currents = zeros(Float64, length(ts))
    
    meas = Float64[]
    currents = _evolve_ratchet!(ρ, ts, expfreq, expM, jϵ, nm, Pjs, meas)
    
    test_dm(ρ, print=false)
    #show(stdout, "text/plain", round.(ρ, digits=2))
    
    return ts, currents
end

#====================================================================================#
#---------- Algorithm That Uses Integer Step Instead of Float Frequency -------------#
#====================================================================================#

function ratchet_measure_int!(ρ0::density_matrix, dt::Float64,
                          T::Float64, nm::Int64,
                          H::Hamiltonian, x::observable, cur::observable, λ::λmatrix;
                          transform=false)
    
    # Start from the initial energy eigenstate with energy n and for a time T
    # evolve with the time-step dt the adjacent flat decay ratchet model with
    # arguments t1, t2, and V. Return the expectation value of the current at 
    # the measured time-steps

    # By default the density matrix is given in energy eigenbasis,
    # to change this use transform flag
    
    ts = [0.0:dt:T; ]
    
        
    # Builds the current and the observable
    # projector struct
    jmom = cur.matrix
    
    Pjs = Projectors(x, H)
    #test_projectors(Pjs, print=false)

    #test_dm(ρ0.matrix, print=false)
    
    # Builds the decay structs
    γs = λsum(λ)
        
    # Builds the Lindbladian structs
    M = λtoM(λ)
    freq = freqarray(H.ens, γs)
    expfreq = freqarray_exp(freq, dt)
    expM = exp(M.matrix*dt)

    jϵ = H.vectors'*jmom*H.vectors 
    transform == true ? (ρ = H.vectors'*ρ0.matrix*H.vectors) : (ρ = ρ0.matrix)
    currents = zeros(Float64, length(ts))
    
    meas = Float64[]
    currents = _evolve_ratchet!(ρ, ts, expfreq, expM, jϵ, nm, Pjs, meas)
    
    #test_dm(ρ, print=false)
    #show(stdout, "text/plain", round.(ρ, digits=2))
    
    return ts, currents
end

#====================================================================================#
#------------------ Total Current Including Charge Displacement ---------------------#
#====================================================================================#

function _evolve_decay_tot!(ρ::Array{ComplexF64}, freqexp::freqarray_exp, expM::Array{Float64}, 
                          Op::Array{ComplexF64})
    
    # Starting from the initial density matrix ρ evolve the density
    # matrix for the time Δt and push the expectation value of the
    # measured observable in the "measured" container
    # THIS CAN BE CHANGED TO ADD THE SAMPLING FREQUENCY OF THE OBSERVABLE
    
    ρ.=freqexp.matrix.*ρ
    ρ[diagind(ρ)]=expM*(ρ[diagind(ρ)])    
    
end

function _measure_ratchet_tot!(ρ::Array{ComplexF64}, Op::Array{ComplexF64}, Pjs::Projectors)
    # The function takes the density matrix and observable projectors as arguments, 
    # calculates the probabilities of different current measurements, and returns
    # a single density matrix projection as an argument 
    
    
    ρbefore = copy(ρ)
    ρ.= zeros(ComplexF64, size(ρbefore)[1], size(ρbefore)[2])
    
    for i in 1:Pjs.number
        ρ .+= Pjs.en_basis[:, :, i]*ρbefore*Pjs.en_basis[:, :, i]
    end
end


function ratchet_cur_tot!(ρ0::density_matrix, dt::Float64,
                          T::Float64, nm::Int64, θ::Float64, ϕ::Float64,
                          H::Hamiltonian, cur::observable, λ::λmatrix;
                          transform=false)
    
    # Start from the initial energy eigenstate with energy n and for a time T
    # evolve with the time-step dt the adjacent flat decay ratchet model with
    # arguments t1, t2, and V. Return the expectation value of the current at 
    # the measured time-steps

    # By default the density matrix is given in energy eigenbasis,
    # to change this use transform flag
    
    ts = [0.0:dt:T; ]
    
    N = Int(H.size//2)
        
    # Builds the current and the observable
    # projector struct
    jmom = cur.matrix
    
    U2 = dm.unitary2R(π/2, π/2+ϕ, θ)
    block = U2*[1.0+0.0im 0.0; 0.0 0.0]*U2'
    θmat = zeros(ComplexF64, 2*N, 2*N)

    for i in 1:2:2*N-1
        θmat[i:i+1, i:i+1] = block
    end

    obs = observable([θmat], (1.0, ), make) 
    
    Pjs = Projectors(obs, H)

    # Builds the decay structs
    γs = λsum(λ)
        
    # Builds the Lindbladian structs
    M = λtoM(λ)
    freq = freqarray(H.ens, γs)
    expfreq = freqarray_exp(freq, dt)
    expM = exp(M.matrix*dt)

    jϵ = H.vectors'*jmom*H.vectors 
    
    Q = dm.measurement_charge_displacement(θ, ϕ, N)
    Qϵ = H.vectors'*Q*H.vectors
    
    
    transform == true ? (ρ = H.vectors'*ρ0.matrix*H.vectors) : (ρ = ρ0.matrix)
    currents_int = zeros(Float64, length(ts))
    
    cur_int = real(tr(ρ*jϵ))
    currents_int[1] = cur_int
    
    for i in 2:length(ts)
        if i%nm != 0
            _evolve_decay_tot!(ρ, expfreq, expM, jϵ)
            cur_int+= real(tr(ρ*jϵ))
            currents_int[i] = cur_int
        else
            Qexp = real(tr(Qϵ*ρ))
            cur_int+=Qexp
            _measure_ratchet_tot!(ρ, jϵ, Pjs)
            currents_int[i] = cur_int 
        end
    end
    
    return ts, currents_int
end

#====================================================================================#
#------------- Algorithm That Uses NEW Projector Construction Method ----------------#
#====================================================================================#

function _measure_ratchetP!(ρ::Array{ComplexF64}, Op::Array{ComplexF64}, Pjsϵ::Array{ComplexF64}, measured::Vector{Float64})
    # The function takes the density matrix and observable projectors as arguments, 
    # calculates the probabilities of different current measurements, and returns
    # a single density matrix projection as an argument 
    
    
    ρbefore = copy(ρ)
    ρ.= zeros(ComplexF64, size(ρbefore)[1], size(ρbefore)[2])
    
    for i in 1:size(Pjsϵ)[3]
        ρ .+= Pjsϵ[:, :, i]*ρbefore*Pjsϵ[:, :, i]
    end
    
    push!(measured, real(tr(Op*ρ)))
end


function _evolve_ratchetP!(ρ::Array{ComplexF64}, ts::Vector{Float64}, expfreq::freqarray_exp,
                          expM::Array{Float64}, jϵ::Array{ComplexF64}, nm::Int64,
                          Pjsϵ::Array{ComplexF64}, measured::Vector{Float64})
    # Helper function that executes one history
    
    push!(measured, real(tr(jϵ*ρ)))
    
    for i in 2:length(ts)
        #show(stdout, "text/plain", round.(ρ, digits=2))
        if i%nm != 0
            _evolve_decay!(ρ, expfreq, expM, jϵ, measured)
        else
            _measure_ratchetP!(ρ, jϵ, Pjsϵ, measured)
        end
    end
    
    return measured
end

function ratchet_measure_intP!(ρ0::density_matrix, dt::Float64,
                          T::Float64, nm::Int64,
                          H::Hamiltonian, θ::Float64, ϕ::Float64, cur::observable, λ::λmatrix;
                          transform=false)
    
    # Start from the initial energy eigenstate with energy n and for a time T
    # evolve with the time-step dt the adjacent flat decay ratchet model with
    # arguments t1, t2, and V. Return the expectation value of the current at 
    # the measured time-steps

    # By default the density matrix is given in energy eigenbasis,
    # to change this use transform flag
    
    ts = [0.0:dt:T; ]
    
        
    # Builds the current and the observable
    # projector struct
    jmom = cur.matrix
    N = cur.size
    
    Pjs, Pjsϵ = measurement_projectors(π/2, ϕ+π/2, θ, Int(N÷2), H.vectors)
    #test_projectors(Pjs, print=false)

    #test_dm(ρ0.matrix, print=false)
    
    # Builds the decay structs
    γs = λsum(λ)
        
    # Builds the Lindbladian structs
    M = λtoM(λ)
    freq = freqarray(H.ens, γs)
    expfreq = freqarray_exp(freq, dt)
    expM = exp(M.matrix*dt)

    jϵ = H.vectors'*jmom*H.vectors 
    transform == true ? (ρ = H.vectors'*ρ0.matrix*H.vectors) : (ρ = ρ0.matrix)
    currents = zeros(Float64, length(ts))
    
    meas = Float64[]
    currents = _evolve_ratchetP!(ρ, ts, expfreq, expM, jϵ, nm, Pjsϵ, meas)
    
    #test_dm(ρ, print=false)
    #show(stdout, "text/plain", round.(ρ, digits=2))
    
    return ts, currents
end

#====================================================================================#
#---------- Integer Algorithm for Measurement Current Expectation Value  ------------#
#====================================================================================#

function _evolve_decay_m!(ρ::Array{ComplexF64}, freqexp::freqarray_exp, expM::Array{Float64}, measured::Vector{Float64})
    
    # Starting from the initial density matrix ρ evolve the density
    # matrix for the time Δt and push the expectation value of the
    # measured observable in the "measured" container
    # THIS CAN BE CHANGED TO ADD THE SAMPLING FREQUENCY OF THE OBSERVABLE
    
    ρ.=freqexp.matrix.*ρ
    ρ[diagind(ρ)]=expM*(ρ[diagind(ρ)])    

end

function _measure_ratchet_m!(ρ::Array{ComplexF64}, Op::Array{ComplexF64}, Pjs::Projectors, measured::Vector{Float64})
    # The function takes the density matrix and observable projectors as arguments, 
    # calculates the probabilities of different current measurements, and returns
    # a single density matrix projection as an argument 
    
    
    ρbefore = copy(ρ)
    jexp = real(tr(Op*ρbefore))
    push!(measured, jexp)
    
    ρ.= zeros(ComplexF64, size(ρbefore)[1], size(ρbefore)[2])

    for i in 1:Pjs.number
        ρ .+= Pjs.en_basis[:, :, i]*ρbefore*Pjs.en_basis[:, :, i]
    end
end


function _evolve_ratchet_m!(ρ::Array{ComplexF64}, ts::Vector{Float64}, expfreq::freqarray_exp,
                          expM::Array{Float64}, jϵ::Array{ComplexF64}, nm::Int64,
                          Pjs::Projectors, measured::Vector{Float64})
    # Helper function that executes one history
    
    for i in 2:length(ts)
        #show(stdout, "text/plain", round.(ρ, digits=2))
        if i%nm != 0
            _evolve_decay_m!(ρ, expfreq, expM, measured)
        else
            _measure_ratchet_m!(ρ, jϵ, Pjs, measured)
        end
    end
    
    return measured
end

function measurement_current(θ::Float64, ϕ::Float64, N::Int64, nm::Int64, dt::Float64)
    
    #Measurement current is defined on the first bond
    
    jm = zeros(ComplexF64, 2*N, 2*N)
    
    mx = sin(θ)*cos(ϕ)
    my = sin(θ)*sin(ϕ)
    mz = cos(θ)
    
    jm[1:2, 1:2] = [1-mz^2 -mz*(mx-1im*my); -mz*(mx+1im*my) mz^2-1]
    
    return N*jm/(2*nm*dt)
end

function measurement_charge_displacement(θ::Float64, ϕ::Float64, N::Int64)
    
    jm = zeros(ComplexF64, 2*N, 2*N)
    
    mx = sin(θ)*cos(ϕ)
    my = sin(θ)*sin(ϕ)
    mz = cos(θ)
    
    jm[1:2, 1:2] = [1-mz^2 -mz*(mx-1im*my); -mz*(mx+1im*my) mz^2-1]
    
    return N*jm/2
end

function ratchet_measure_int_m!(ρ0::density_matrix, dt::Float64,
                          T::Float64, nm::Int64, θ::Float64, ϕ::Float64, N::Int64,
                          H::Hamiltonian, obs::observable, λ::λmatrix;
                          transform=false)
    
    # Start from the initial energy eigenstate with energy n and for a time T
    # evolve with the time-step dt the adjacent flat decay ratchet model with
    # arguments t1, t2, and V. Return the expectation value of the current at 
    # the measured time-steps

    # By default the density matrix is given in energy eigenbasis,
    # to change this use transform flag
    
    ts = [0.0:dt:T; ]
    
    jm = measurement_current(θ, ϕ, N, nm, dt)
        
    # Builds the current and the observable
    # projector struct
    
    Pjs = Projectors(obs, H)
    #test_projectors(Pjs, print=false)

    #test_dm(ρ0.matrix, print=false)
    
    # Builds the decay structs
    γs = λsum(λ)
        
    # Builds the Lindbladian structs
    M = λtoM(λ)w
    freq = freqarray(H.ens, γs)
    expfreq = freqarray_exp(freq, dt)
    expM = exp(M.matrix*dt)
 
    jϵ = H.vectors'*jm*H.vectors
    transform == true ? (ρ = H.vectors'*ρ0.matrix*H.vectors) : (ρ = ρ0.matrix)
    currents = zeros(Float64, length(ts))
    
    meas = Float64[]
    currents = _evolve_ratchet_m!(ρ, ts, expfreq, expM, jϵ, nm, Pjs, meas)
    
    #test_dm(ρ, print=false)
    #show(stdout, "text/plain", round.(ρ, digits=2))
    
    return ts, currents
end

#=====================================================================#
#--------------------- Steady State Algorithm ------------------------#
#=====================================================================#

function _evolve_decay_steady!(ρ::Array{ComplexF64}, freqexp::freqarray_exp, expM::Array{Float64}, 
                          Op::Array{ComplexF64}, measured::Vector{Float64}, measure::Bool)
    
    # Starting from the initial density matrix ρ evolve the density
    # matrix for the time Δt and push the expectation value of the
    # measured observable in the "measured" container
    # THIS CAN BE CHANGED TO ADD THE SAMPLING FREQUENCY OF THE OBSERVABLE
    
    ρ.=freqexp.matrix.*ρ
    ρ[diagind(ρ)]=expM*(ρ[diagind(ρ)])    
    
    if measure==true
        push!(measured, real(tr(Op*ρ)))
    end
end

function _measure_ratchet_steady!(ρ::Array{ComplexF64}, Op::Array{ComplexF64}, Pjs::Projectors, measured::Vector{Float64}, measure::Bool)
    # The function takes the density matrix and observable projectors as arguments, 
    # calculates the probabilities of different current measurements, and returns
    # a single density matrix projection as an argument 
    
    
    ρbefore = copy(ρ)
    ρ.= zeros(ComplexF64, size(ρbefore)[1], size(ρbefore)[2])
    
    for i in 1:Pjs.number
        ρ .+= Pjs.en_basis[:, :, i]*ρbefore*Pjs.en_basis[:, :, i]
    end
    
    if measure==true 
        push!(measured, real(tr(Op*ρ)))
    end
end


function _evolve_ratchet_steady!(ρ::Array{ComplexF64}, ts::Vector{Float64}, expfreq::freqarray_exp,
                          expM::Array{Float64}, jϵ::Array{ComplexF64}, nm::Int64,
                          Pjs::Projectors, measured::Vector{Float64}, fraction::Float64)
    # Helper function that executes one history
    
    measure=false
    
    for i in 2:length(ts)
        #show(stdout, "text/plain", round.(ρ, digits=2))
        if i >= fraction*length(ts)
           measure=true
        end
        
        if i%nm != 0
            _evolve_decay_steady!(ρ, expfreq, expM, jϵ, measured, measure)
        else
            _measure_ratchet_steady!(ρ, jϵ, Pjs, measured, measure)
        end
    end
    
    return measured
end


function ratchet_measure_steady!(ρ0::density_matrix, dt::Float64,
                          T::Float64, nm::Int64,
                          H::Hamiltonian, x::observable, cur::observable, λ::λmatrix;
                          fraction=0.9, transform=false)
    
    # Start from the initial energy eigenstate with energy n and for a time T
    # evolve with the time-step dt the adjacent flat decay ratchet model with
    # arguments t1, t2, and V. Return the expectation value of the current at 
    # the measured time-steps

    # By default the density matrix is given in energy eigenbasis,
    # to change this use transform flag
    
    ts = [0.0:dt:T; ]
    tsteady = [fraction*T:dt:T; ]
    
        
    # Builds the current and the observable
    # projector struct
    jmom = cur.matrix
    
    Pjs = Projectors(x, H)
    #test_projectors(Pjs, print=false)

    #test_dm(ρ0.matrix, print=false)
    
    # Builds the decay structs
    γs = λsum(λ)
        
    # Builds the Lindbladian structs
    M = λtoM(λ)
    freq = freqarray(H.ens, γs)
    expfreq = freqarray_exp(freq, dt)
    expM = exp(M.matrix*dt)

    jϵ = H.vectors'*jmom*H.vectors 
    transform == true ? (ρ = H.vectors'*ρ0.matrix*H.vectors) : (ρ = ρ0.matrix)
    currents = zeros(Float64, length(ts))
    
    meas = Float64[]
    currents = _evolve_ratchet_steady!(ρ, ts, expfreq, expM, jϵ, nm, Pjs, meas, fraction)
    
    #test_dm(ρ, print=false)
    #show(stdout, "text/plain", round.(ρ, digits=2))
    
    return tsteady, currents
end





#====================================================================================#
#------------------------ General Algorithm for Two Qubits --------------------------#
#====================================================================================#

function _measure_ratchet_twoqubit!(ρ::Array{ComplexF64}, Op::Array{ComplexF64},
                                    Pjs, measured::Vector{Float64})
    # The function takes the density matrix and observable projectors as arguments, 
    # calculates the probabilities of different current measurements, and returns
    # a single density matrix projection as an argument 
    
    
    for i in 1:Pjs.number
        nonP = diagm(ones(ComplexF64, size(ρ)[1])).-Pjs.en_basis[:, :, i]
        ρ .= Pjs.en_basis[:, :, i]*ρ*Pjs.en_basis[:, :, i].+nonP*ρ*nonP
    end
    
    push!(measured, real(tr(Op*ρ)))
end


function _evolve_ratchet_twoqubit!(ρ::Array{ComplexF64}, ts::Vector{Float64}, expfreq::freqarray_exp,
                          expM::Array{Float64}, jϵ::Array{ComplexF64}, nm::Int64,
                          Pjs, measured::Vector{Float64})
    # Helper functions that executes one history
    
    push!(measured, real(tr(jϵ*ρ)))
    
    for i in 2:length(ts)
        #show(stdout, "text/plain", round.(ρ, digits=2))
        if i%nm != 0
            _evolve_decay!(ρ, expfreq, expM, jϵ, measured)
        else
            _measure_ratchet_twoqubit!(ρ, jϵ, Pjs, measured)
        end
    end
    
    return measured
end


function ratchet_measure_twoqubit!(ρ0::density_matrix, dt::Float64,
                          T::Float64, ωm::Float64,
                          H::Hamiltonian, control::Array{ComplexF64},
                          cur::observable, λ::λmatrix, site_num::Int64; transform=false, measure_one=false)
    
    # Start from the initial energy eigenstate with energy n and for a time T
    # evolve with the time-step dt the adjacent flat decay ratchet model with
    # arguments t1, t2, and V. Return the expectation value of the current at 
    # the measured time-steps

    # By default the density matrix is given in energy eigenbasis,
    # to change this use transform flag
    
    ts = [0.0:dt:T; ]
    
    # Builds the current and the observable
    # projector struct
    jmom = cur.matrix
    measure_one==false ? (Pjs = ProjectorsTwo(control, H, site_num)) : (Pjs=ProjectorsOcc(H, control, site_num))
    #test_projectors(Pjs, print=false)
    #println("\nFIRST:\n")
    #show(stdout, "text/plain", round.(Pjs.matrices[:,:, 1], digits=2))
    #println("\nSECOND:\n")
    #show(stdout, "text/plain", round.(Pjs.matrices[:,:, 2], digits=2))
    test_dm(ρ0.matrix, print=false)
    
    ωm != 0.0 ? (nm = Int(1÷(dt*ωm))) : (nm=typemax(Int))
    
    # Builds the decay structs
    γs = λsum(λ)
        
    # Builds the Lindbladian structs
    M = λtoM(λ)
    freq = freqarray(H.ens, γs)
    expfreq = freqarray_exp(freq, dt)
    expM = exp(M.matrix*dt)

    jϵ = H.vectors'*jmom*H.vectors 
    transform == true ? (ρ = H.vectors'*ρ0.matrix*H.vectors) : (ρ = ρ0.matrix)
    currents = zeros(Float64, length(ts))
    
    meas = Float64[]
    currents = _evolve_ratchet_twoqubit!(ρ, ts, expfreq, expM, jϵ, nm, Pjs, meas)
    
    #show(stdout, "text/plain", round.(ρ, digits=2))
    
    return ts, currents
end

#==========================================================================#
#------------------------ Imperfect Measurements --------------------------#
#==========================================================================#

function _measure_ratchet_povm!(ρ::Array{ComplexF64}, Op::Array{ComplexF64},
                                Ms::Kraus, measured::Vector{Float64}; single_site=false)
    # The function takes the density matrix and observable projectors as arguments, 
    # calculates the probabilities of different current measurements, and returns
    # a single density matrix projection as an argument 
    
    ρbefore = copy(ρ)
    ρ.= zeros(ComplexF64, size(ρbefore)[1], size(ρbefore)[2])
        
    if single_site==false
        for i in 1:Int(Ms.number)
            ρ .+= Ms.en_basis[:, :, i]*ρbefore*Ms.en_basis[:, :, i]'
        end
    elseif single_site==true
        ρ .+= Ms.en_basis[:, :, 1]*ρbefore*Ms.en_basis[:, :, 1]'
        ρ .+= Ms.en_basis[:, :, Ms.number]*ρbefore*Ms.en_basis[:, :, Ms.number]'
    else
       error("Single site can be only true or false.") 
    end    
    push!(measured, real(tr(Op*ρ)))
end


function _evolve_ratchet_povm!(ρ::Array{ComplexF64}, ts::Vector{Float64}, expfreq::freqarray_exp,
                          expM::Array{Float64}, jϵ::Array{ComplexF64}, nm::Int64,
                          Ms::Kraus, measured::Vector{Float64}; single_site=false)
    # Helper functions that executes one history
    
    push!(measured, real(tr(jϵ*ρ)))
    
    for i in 2:length(ts)
        #show(stdout, "text/plain", round.(ρ, digits=2))
        if i%nm != 0
            _evolve_decay!(ρ, expfreq, expM, jϵ, measured)
        else
            _measure_ratchet_povm!(ρ, jϵ, Ms, measured)
        end
    end
    
    return measured
end


function ratchet_measure_povm!(ρ0::density_matrix, dt::Float64,
                          T::Float64, ωm::Float64, ϕ::Float64, b::Float64, Ψ::Float64,
                          H::Hamiltonian, control::Array{ComplexF64}, x::observable, cur::observable, λ::λmatrix;
                          transform=false, single_site=false)
    
    # Performs the measurement of an observable either at each bond or
    # on a single site, such that each outcome of the measurement has some uncertainty;
    # outcome i implies the projection on the i-th eigenstate with probability cos^2(\phi)
    # projection on (i+1)-th eigenstate with probability sqrt(sin^2(\phi)+b) and on
    # (i-1)-th eigenstate with the probability sqrt(sin^2(\phi)-b)
    
    ts = [0.0:dt:T; ]
    
        
    # Builds the current and the observable
    # projector struct
    jmom = cur.matrix
    
    # Build projectors and use them to build
    # Kraus operators
    Pjs = Projectors(x, H)
    test_projectors(Pjs, print=false)
    Ms = Kraus(Pjs, H, control, ϕ, b, Ψ; single_site=single_site)

    test_dm(ρ0.matrix, print=false)
    
    ωm != 0.0 ? (nm = Int(1÷(dt*ωm))) : (nm=typemax(Int))
    
    # Builds the decay structs
    γs = λsum(λ)
        
    # Builds the Lindbladian structs
    M = λtoM(λ)
    freq = freqarray(H.ens, γs)
    expfreq = freqarray_exp(freq, dt)
    expM = exp(M.matrix*dt)

    jϵ = H.vectors'*jmom*H.vectors 
    transform == true ? (ρ = H.vectors'*ρ0.matrix*H.vectors) : (ρ = ρ0.matrix)
    currents = zeros(Float64, length(ts))
    
    meas = Float64[]
    currents = _evolve_ratchet_povm!(ρ, ts, expfreq, expM, jϵ, nm, Ms, meas; single_site=single_site)
    
    #test_dm(ρ, print=false)
    #show(stdout, "text/plain", round.(ρ, digits=2))
    
    return ts, currents
end



function _measure_ratchet_imperfect!(ρ::Array{ComplexF64}, Op::Array{ComplexF64},
                                    Ms::KrausOperators, measured::Vector{Float64}; skip=false)
    # The function takes the density matrix and observable projectors as arguments, 
    # calculates the probabilities of different current measurements, and returns
    # a single density matrix projection as an argument 
    
    ρbefore = copy(ρ)
    ρ.= zeros(ComplexF64, size(ρbefore)[1], size(ρbefore)[2])
        
    if skip==false
        for i in 1:Int(Ms.number-1)
            ρ .+= Ms.en_basis[:, :, i]*ρbefore*Ms.en_basis[:, :, i]'
        end
    elseif skip=="b"
        for i in 1:Int((Ms.number-1) ÷ 2)
            ρ .+= Ms.en_basis[:, :, 2*i-1]*ρbefore*Ms.en_basis[:, :, 2*i-1]'
        end
        ρ .+= Ms.en_basis[:, :, Ms.number]*ρbefore*Ms.en_basis[:, :, Ms.number]'
    elseif skip=="a"
        for i in 1:Int((Ms.number-1) ÷ 2)
            ρ .+= Ms.en_basis[:, :, 2*i]*ρbefore*Ms.en_basis[:, :, 2*i]'
        end
        ρ .+= Ms.en_basis[:, :, Ms.number]*ρbefore*Ms.en_basis[:, :, Ms.number]'
    else
       error("skip can only be a, b, or false.") 
    end
        
        
    push!(measured, real(tr(Op*ρ)))
end


function _evolve_ratchet_imperfect!(ρ::Array{ComplexF64}, ts::Vector{Float64}, expfreq::freqarray_exp,
                          expM::Array{Float64}, jϵ::Array{ComplexF64}, nm::Int64,
                          Ms::KrausOperators, measured::Vector{Float64}; skip=false)
    # Helper functions that executes one history
    
    push!(measured, real(tr(jϵ*ρ)))
    
    for i in 2:length(ts)
        #show(stdout, "text/plain", round.(ρ, digits=2))
        if i%nm != 0
            _evolve_decay!(ρ, expfreq, expM, jϵ, measured)
        else
            _measure_ratchet_imperfect!(ρ, jϵ, Ms, measured; skip=skip)
        end
    end
    
    return measured
end

function ratchet_measure_imperfect!(ρ0::density_matrix, dt::Float64,
                          T::Float64, ωm::Float64, ξ::Float64,
                          H::Hamiltonian, control::Array{ComplexF64},
                          cur::observable, λ::λmatrix, site_num::Int64; transform=false, skip=false)
    
    # Start from the initial energy eigenstate with energy n and for a time T
    # evolve with the time-step dt the adjacent flat decay ratchet model with
    # arguments t1, t2, and V. Return the expectation value of the current at 
    # the measured time-steps

    # By default the density matrix is given in energy eigenbasis,
    # to change this use transform flag
    
    ts = [0.0:dt:T; ]
    
    # Builds the current and the observable
    # projector struct
    jmom = cur.matrix
    Ms = KrausOperators(control, H, site_num, ξ)
    #test_projectors(Pjs, print=false)
    
    test_dm(ρ0.matrix, print=false)
    
    ωm != 0.0 ? (nm = Int(1÷(dt*ωm))) : (nm=typemax(Int))
    
    # Builds the decay structs
    γs = λsum(λ)
        
    # Builds the Lindbladian structs
    M = λtoM(λ)
    freq = freqarray(H.ens, γs)
    expfreq = freqarray_exp(freq, dt)
    expM = exp(M.matrix*dt)

    jϵ = H.vectors'*jmom*H.vectors 
    transform == true ? (ρ = H.vectors'*ρ0.matrix*H.vectors) : (ρ = ρ0.matrix)
    currents = zeros(Float64, length(ts))
    
    meas = Float64[]
    currents = _evolve_ratchet_imperfect!(ρ, ts, expfreq, expM, jϵ, nm, Ms, meas; skip=skip)
    
    #show(stdout, "text/plain", round.(ρ, digits=2))
    
    return ts, currents
end

#This function is not working!!!
#Or fluctuations_2band function (???)

function ratchet_fluctuations!(ρ0::density_matrix, dt::Float64,
                               T::Float64, ωm::Float64, H::Hamiltonian, x::observable, 
                               cur::observable, cur2::observable, λ::λmatrix)
    
    # Start from the initial energy eigenstate with energy n and for a time T
    # evolve with the time-step dt the adjacent flat decay ratchet model with
    # arguments t1, t2, and V. Return the expectation value of the current at 
    # the measured time-steps
    
    ts = [0.0:dt:T; ]
    
    # Builds the current and the observable
    # projector struct
    jmom = cur.matrix
    jmom2 = cur2.matrix
    
    Pjs = Projectors(x, H)
    test_projectors(Pjs, print=false)
    test_dm(ρ0.matrix, print=false)
    
    ωm != 0.0 ? (nm = Int(1÷(dt*ωm))) : (nm=typemax(Int))

    # Builds the decay structs
    γs = λsum(λ)
        
    # Builds the Lindbladian structs
    M = λtoM(λ)
    freq = freqarray(H.ens, γs)
    expfreq = freqarray_exp(freq, dt)
    expM = exp(M.matrix*dt)

    jϵ = H.vectors'*jmom*H.vectors 
    jϵ2 = H.vectors'*jmom2*H.vectors    

    ρ1 = copy(ρ0.matrix)
    ρ2 = copy(ρ0.matrix)
    currents = zeros(Float64, length(ts))
    currents_squared = zeros(Float64, length(ts))
    
    meas1 = Float64[]
    meas2 = Float64[]
    
    currents = _evolve_ratchet!(ρ1, ts, expfreq, expM, jϵ, nm, Pjs, meas1)
    currents_squared = _evolve_ratchet!(ρ2, ts, expfreq, expM, jϵ2, nm, Pjs, meas2)
    
    for i in 1:length(currents)
        @assert currents_squared[i]-(currents.^2)[i] >= 0.0 "The value currents_squared is $(currents_squared[i]), and currents squared=$((currents.^2)[i]) for $(i), $(H.args), ωm=$(ωm), γ=$(λ.matrix[1, 2]), jeps^2=$(sum(jϵ*jϵ)), jeps2=$(sum(jϵ2))"
    end
 
    return ts, sqrt.(currents_squared.-currents.^2)
end


####################################################################################
###################### Observable Parameter Dependence #############################
####################################################################################

function dc_current(ts::Vector{Float64}, cur::Vector{Float64})
    
    m(t, p) = p[1].+p[2] * t
    p0 = [0.0, 0.02]
    dt = ts[2]-ts[1]
    
    fit = curve_fit(m, ts, cumsum(cur), p0)
    
    param = fit.param
    
    param[2] = dt*param[2]
    
    return param
end


function current_spectrum(ts::Vector{Float64}, cur::Vector{Float64})
    # Takes the times and current, and returns the Fourier transform 
    # for the second half of the signal
    
    tcut, curcut = ts[Int(length(ts)÷2):end], cur[Int(length(ts)÷2):end]
    dt = tcut[2]-tcut[1]
    T = curcut[length(curcut)]-curcut[1]
    
    freqs = fftfreq(length(tcut), 1/dt)
    curfft = 2.0*fft(curcut)/length(tcut)
    
    return freqs, curfft
end


function DCcur(ts::Vector{Float64}, cur::Vector{Float64})

    DC = cumsum(cur)[length(cur)]-cur[1]
    
    return DC/length(ts)
end

#=====================================================================#
#-------------------------- Two Band Model ---------------------------#
#=====================================================================#

function parameter_draw(t1::Float64, t2::Float64, V::Float64, γ::Float64,
                        N::Int64, ωm::Float64, dt::Float64, T::Float64,
                        paras::Vector{Float64}; parameter="V", n=1, d="A", energy=false)
    # Draw the current dependence for
    # a specific parameter choice in some range
    
    ts = [0.0:dt:T; ]
    dcs = Vector{Float64}(undef, length(paras))

    if parameter=="V"
        for i in eachindex(paras)
            ρ0, H, x, J, λ = build_objects_2band(t1, t2, paras[i], γ, N, n; decay=d, en=energy)
            ts, cur = ratchet_measure!(ρ0, dt, T, ωm, H, x, J, λ)
            dcs[i] = dc_current(ts[Int(length(ts)÷2):end], cur[Int(length(ts)÷2):end])[2]
        end
    elseif parameter=="t1"
        for i in eachindex(paras)
            ρ0, H, x, J, λ = build_objects_2band(paras[i], t2, V, γ, N, n; decay=d, en=energy)
            ts, cur = ratchet_measure!(ρ0, dt, T, ωm, H, x, J, λ)
            dcs[i] = dc_current(ts[Int(length(ts)÷2):end], cur[Int(length(ts)÷2):end])[2]
        end
    elseif parameter=="t2"
        for i in eachindex(paras)
            ρ0, H, x, J, λ = build_objects_2band(t1, paras[i], V, γ, N, n; decay=d, en=energy)
            ts, cur = ratchet_measure!(ρ0, dt, T, ωm, H, x, J, λ)
            dcs[i] = dc_current(ts[Int(length(ts)÷2):end], cur[Int(length(ts)÷2):end])[2]
        end
    elseif parameter=="γ"
        for i in eachindex(paras)
            ρ0, H, x, J, λ = build_objects_2band(t1, t2, V, paras[i], N, n; decay=d, en=energy)
            ts, cur = ratchet_measure!(ρ0, dt, T, ωm, H, x, J, λ)
            dcs[i] = dc_current(ts[Int(length(ts)÷2):end], cur[Int(length(ts)÷2):end])[2]
        end
    elseif parameter=="ωm"
        for i in eachindex(paras)
            ρ0, H, x, J, λ = build_objects_2band(t1, t2, V, γ, N, n; decay=d, en=energy)
            ts, cur = ratchet_measure!(ρ0, dt, T, paras[i], H, x, J, λ)
            dcs[i] = dc_current(ts[Int(length(ts)÷2):end], cur[Int(length(ts)÷2):end])[2]
        end
    else
        error("The parameter does not exist")
    end
    
    return paras, dcs
end

function temperature_dependence(t1::Float64, t2::Float64, V::Float64, γ::Float64,
                        N::Int64, ωm::Float64, dt::Float64, T::Float64,
                        Vs::Vector{Float64}, temperature::Float64; obs="xJ", n=1, site="a", energy=false)
    
    # This function takes an array of values of potential tilt, and temperature
    # and returns the DC currents as a function of the potential V
    
    ts = [0.0:dt:T; ]
    dcs = Vector{Float64}(undef, length(Vs))
    
    if obs == "xJ"
        for i in eachindex(Vs)
            ρ0, H, x, J, λ = build_objects_2band(t1, t2, Vs[i], γ, N, n; decay="T", temp=temperature, en=energy)
            ts, cur = ratchet_measure!(ρ0, dt, T, ωm, H, x, J, λ)
            dcs[i] = dc_current(ts[Int(length(ts)÷2):end], cur[Int(length(ts)÷2):end])[2]
        end        
    elseif obs == "JJ"
        for i in eachindex(Vs)
            ρ0, H, x, J, λ = build_objects_2band(t1, t2, Vs[i], γ, N, n; decay="T", temp=temperature, en=energy)
            ts, cur = ratchet_measure!(ρ0, dt, T, ωm, H, J, J, λ)
            dcs[i] = dc_current(ts[Int(length(ts)÷2):end], cur[Int(length(ts)÷2):end])[2]
        end
    elseif obs == "NJ"
        for i in eachindex(Vs)
            ρ0, H, x, J, λ = build_objects_2band(t1, t2, Vs[i], γ, N, n; decay="T", temp=temperature, en=energy)
            if site=="a"
                Occ = build_occupation(N, site="a")
            elseif site=="b"
                Occ = build_occupation(N, site="b")
            else
                error("There are only sites a and b.")
            end
            ts, cur = ratchet_measure!(ρ0, dt, T, ωm, H, Occ, J, λ)
            dcs[i] = dc_current(ts[Int(length(ts)÷2):end], cur[Int(length(ts)÷2):end])[2]
        end
    else
        error("The combination of observables doesn't exist")
    end
    return Vs, dcs
end

function fluctuations_2band(t1::Float64, t2::Float64, V::Float64, γ::Float64,
                        N::Int64, ωm::Float64, dt::Float64, T::Float64,
                        Vs::Vector{Float64}, temperature::Float64; obs="xJ", n=1, energy=false)
    
    # This function takes an array of values of potential tilt, and temperature
    # and returns the DC currents as a function of the potential V
    
    ts = [0.0:dt:T; ]
    Δdcs = Vector{Float64}(undef, length(Vs))
    
    if obs == "xJ"
        for i in eachindex(Vs)
            ρ0, H, x, J, λ = build_objects_2band(t1, t2, Vs[i], γ, N, n; decay="T", temp=temperature, en=energy)
            J2 = build_squares_2band(t1, t2, N)
            ts, Δcur = ratchet_fluctuations!(ρ0, dt, T, ωm, H, x, J, J2, λ)
            Δdcs[i] = dc_current(ts[Int(length(ts)÷2):end],  Δcur[Int(length(ts)÷2):end])[2]
        end        
    elseif obs == "JJ"
        for i in eachindex(Vs)
            ρ0, H, x, J, λ = build_objects_2band(t1, t2, Vs[i], γ, N, n; decay="T", temp=temperature, en=energy)
            J2 = build_squares_2band(t1, t2, N)
            ts, Δcur = ratchet_fluctuations!(ρ0, dt, T, ωm, H, J, J, J2, λ)
            Δdcs[i] = dc_current(ts[Int(length(ts)÷2):end], Δcur[Int(length(ts)÷2):end])[2]
        end        
    else
        error("The combination of observables doesn't exist")
    end
    
    return Vs, Δdcs
end

function occupation_dependence(t1::Float64, t2::Float64, V::Float64, γ::Float64,
                        N::Int64, ωm::Float64, dt::Float64, T::Float64,
                        Vs::Vector{Float64}, temperature::Float64; obs="bb", n=1, energy=false)
    # This function measures occupation of sites given by the first index in obs
    # and returns the expectation values of occupation of sites given by the second
    # index in obs
    
    ts = [0.0:dt:T; ]
    dcs = Vector{Float64}(undef, length(Vs))
    
    for i in eachindex(Vs)
        ρ0, H, x, J, λ = build_objects_2band(t1, t2, Vs[i], γ, N, n; decay="T", temp=temperature, en=energy)
        N1 = build_occupation(N, site=string(obs[1]))
        N2 = build_occupation(N, site=string(obs[2]))
        ts, cur = ratchet_measure!(ρ0, dt, T, ωm, H, N1, N2, λ)
        dcs[i] = dc_current(ts[Int(length(ts)÷2):end], cur[Int(length(ts)÷2):end])[2]
    end
    
    return Vs, dcs 
end

#=====================================================================#
#-------------------------- Three Band Model -------------------------#
#=====================================================================#

function parameter_draw3(t1::Float64, t2::Float64, t3::Float64, V1::Float64, 
                        V2::Float64, γ::Float64, N::Int64, ωm::Float64,
                        dt::Float64, T::Float64, paras::Vector{Float64};
                        parameter="V", n=1, d="A", energy=false)
    # Draw the current dependence for
    # a specific parameter choice in some range
    
    ts = [0.0:dt:T; ]
    dcs = Vector{Float64}(undef, length(paras))

    if parameter=="t1"
        for i in eachindex(paras)
            ρ0, H, x, J, λ = build_objects_3band(paras[i], t2, t3, V1, V2, γ, N, n; decay=d, en=energy)
            ts, cur = ratchet_measure!(ρ0, dt, T, ωm, H, x, J, λ)
            dcs[i] = dc_current(ts[Int(length(ts)÷2):end], cur[Int(length(ts)÷2):end])[2]
        end
    elseif parameter=="t2"
        for i in eachindex(paras)
            ρ0, H, x, J, λ = build_objects_3band(t1, paras[i], t3, V1, V2, γ, N, n; decay=d, en=energy)
            ts, cur = ratchet_measure!(ρ0, dt, T, ωm, H, x, J, λ)
            dcs[i] = dc_current(ts[Int(length(ts)÷2):end], cur[Int(length(ts)÷2):end])[2]
        end
    elseif parameter=="t3"
        for i in eachindex(paras)
            ρ0, H, x, J, λ = build_objects_3band(t1, t2, paras[i], V1, V2, γ, N, n; decay=d, en=energy)
            ts, cur = ratchet_measure!(ρ0, dt, T, ωm, H, x, J, λ)
            dcs[i] = dc_current(ts[Int(length(ts)÷2):end], cur[Int(length(ts)÷2):end])[2]
        end
    elseif parameter=="V"
        for i in eachindex(paras)
            ρ0, H, x, J, λ = build_objects_3band(t1, t2, t3, paras[i], -paras[i], γ, N, n; decay=d, en=energy)
            ts, cur = ratchet_measure!(ρ0, dt, T, ωm, H, J, J, λ)
            dcs[i] = dc_current(ts[Int(length(ts)÷2):end], cur[Int(length(ts)÷2):end])[2]
        end
    elseif parameter=="Vb"
        for i in eachindex(paras)
            ρ0, H, x, J, λ = build_objects_3band(t1, t2, t3, paras[i], -paras[i], γ, N, n; decay=d, en=energy)
            ts, cur = ratchet_measure!(ρ0, dt, T, ωm, H, x, J, λ)
            dcs[i] = DCcur(ts, cur)
        end
    elseif parameter=="V1"
        for i in eachindex(paras)
            ρ0, H, x, J, λ = build_objects_3band(t1, t2, t3, paras[i], V2, γ, N, n; decay=d, en=energy)
            ts, cur = ratchet_measure!(ρ0, dt, T, ωm, H, x, J, λ)
            dcs[i] = dc_current(ts[Int(length(ts)÷2):end], cur[Int(length(ts)÷2):end])[2]
        end
    elseif parameter=="V2"
        for i in eachindex(paras)
            ρ0, H, x, J, λ = build_objects_3band(t1, t2, t3, V1, paras[i], γ, N, n; decay=d, en=energy)
            ts, cur = ratchet_measure!(ρ0, dt, T, ωm, H, x, J, λ)
            dcs[i] = dc_current(ts[Int(length(ts)÷2):end], cur[Int(length(ts)÷2):end])[2]
        end
    elseif parameter=="γ"
        for i in eachindex(paras)
            ρ0, H, x, J, λ = build_objects_3band(t1, t2, t3, V1, V2, paras[i], N, n; decay=d, en=energy)
            ts, cur = ratchet_measure!(ρ0, dt, T, ωm, H, x, J, λ)
            dcs[i] = dc_current(ts[Int(length(ts)÷2):end], cur[Int(length(ts)÷2):end])[2]
        end
    elseif parameter=="ωm"
        for i in eachindex(paras)
            ρ0, H, x, J, λ = build_objects_3band(t1, t2, t3, V1, V2, γ, N, n; decay=d, en=energy)
            ts, cur = ratchet_measure!(ρ0, dt, T, paras[i], H, x, J, λ)
            dcs[i] = dc_current(ts[Int(length(ts)÷2):end], cur[Int(length(ts)÷2):end])[2]
        end
    else
        error("The parameter does not exist")
    end
    
    return paras, dcs
end

function temperature_dependence3(t1::Float64, t2::Float64, t3::Float64, 
                        V1::Float64, V2::Float64, γ::Float64, N::Int64,
                        ωm::Float64, dt::Float64, T::Float64, Vs::Vector{Float64},
                        temperature::Float64; obs="xJ", n=1, site="a", energy=false)
    
    # This function takes an array of values of potential tilt, and temperature
    # and returns the DC currents as a function of the potential V
    
    ts = [0.0:dt:T; ]
    dcs = Vector{Float64}(undef, length(Vs))
    
    if obs == "xJ"
        for i in eachindex(Vs)
            ρ0, H, x, J, λ = build_objects_3band(t1, t2, t3, Vs[i], -Vs[i], γ, N, n; decay="T", temp=temperature, en=energy)
            ts, cur = ratchet_measure!(ρ0, dt, T, ωm, H, x, J, λ)
            dcs[i] = dc_current(ts[Int(length(ts)÷2):end], cur[Int(length(ts)÷2):end])[2]
        end        
    elseif obs == "JJ"
        for i in eachindex(Vs)
            ρ0, H, x, J, λ = build_objects_3band(t1, t2, t3, Vs[i], -Vs[i], γ, N, n; decay="T", temp=temperature, en=energy)
            ts, cur = ratchet_measure!(ρ0, dt, T, ωm, H, J, J, λ)
            dcs[i] = dc_current(ts[Int(length(ts)÷2):end], cur[Int(length(ts)÷2):end])[2]
        end   
    elseif obs == "NJ"
        for i in eachindex(Vs)
            ρ0, H, x, J, λ = build_objects_3band(t1, t2, t3, Vs[i], -Vs[i], γ, N, n; decay="T", temp=temperature, en=energy)
            if site=="a"
                Occ = build_occupation3(N, site="a")
            elseif site=="b"
                Occ = build_occupation3(N, site="b")
            elseif site=="c"
                Occ = build_occupation3(N, site="c")
            else
                error("There are only sites a, b, and c.")
            end
            ts, cur = ratchet_measure!(ρ0, dt, T, ωm, H, Occ, J, λ)
            dcs[i] = dc_current(ts[Int(length(ts)÷2):end], cur[Int(length(ts)÷2):end])[2]
        end
    else
        error("The combination of observables doesn't exist")
    end
    
    return Vs, dcs
end

function fluctuations_3band(t1::Float64, t2::Float64, t3::Float64, V1::Float64, V2::Float64, 
                            γ::Float64, N::Int64, ωm::Float64, dt::Float64, T::Float64,
                            Vs::Vector{Float64}, temperature::Float64; obs="xJ", n=1, energy=false)
    
    # This function takes an array of values of potential tilt, and temperature
    # and returns the DC currents as a function of the potential V
    
    ts = [0.0:dt:T; ]
    Δdcs = Vector{Float64}(undef, length(Vs))
    
    if obs == "xJ"
        for i in eachindex(Vs)
            ρ0, H, x, J, λ = build_objects_3band(t1, t2, t3, Vs[i], -Vs[i], γ, N, n; decay="T", temp=temperature, en=energy)
            J2 = build_squares_3band(t1, t2, t3, N)
            ts, Δcur = ratchet_fluctuations!(ρ0, dt, T, ωm, H, x, J, J2, λ)
            Δdcs[i] = dc_current(ts[Int(length(ts)÷2):end],  Δcur[Int(length(ts)÷2):end])[2]
        end        
    elseif obs == "JJ"
        for i in eachindex(Vs)
            ρ0, H, x, J, λ = build_objects_3band(t1, t2, t3, Vs[i], -Vs[i], γ, N, n; decay="T", temp=temperature, en=energy)
            J2 = build_squares_3band(t1, t2, t3, N)
            ts, Δcur = ratchet_fluctuations!(ρ0, dt, T, ωm, H, J, J, J2, λ)
            Δdcs[i] = dc_current(ts[Int(length(ts)÷2):end], Δcur[Int(length(ts)÷2):end])[2]
        end        
    else
        error("The combination of observables doesn't exist")
    end
    
    return Vs, Δdcs
end

function occupation_dependence3(t1::Float64, t2::Float64, t3::Float64, V1::Float64,
                                V2::Float64, γ::Float64, N::Int64, ωm::Float64, dt::Float64,
                                T::Float64, Vs::Vector{Float64}, temperature::Float64; obs="cc", n=1, energy=false)
    # This function measures occupation of sites given by the first index in obs
    # and returns the expectation values of occupation of sites given by the second
    # index in obs
    
    ts = [0.0:dt:T; ]
    dcs = Vector{Float64}(undef, length(Vs))
    
    for i in eachindex(Vs)
        ρ0, H, x, J, λ = build_objects_3band(t1, t2, t3, Vs[i], -Vs[i], γ, N, n; decay="T", temp=temperature, en=energy)
        N1 = build_occupation3(N, site=string(obs[1]))
        N2 = build_occupation3(N, site=string(obs[2]))
        ts, cur = ratchet_measure!(ρ0, dt, T, ωm, H, N1, N2, λ)
        dcs[i] = dc_current(ts[Int(length(ts)÷2):end], cur[Int(length(ts)÷2):end])[2]
    end
    
    return Vs, dcs 
end


####################################################################################
################################## TESTS ###########################################
####################################################################################
    
function test_projectors(Pjs::Projectors; print=true)
    
    Ps = [Pjs.matrices[:, :, i] for i in 1:Pjs.number]
    Psϵ = [Pjs.en_basis[:, :, i] for i in 1:Pjs.number]
    N = size(Ps[1])[1]
    
    @assert isapprox(tr(sum(Ps)), N, atol=1e-7) "Not closed tr(sum(Ps))=$(tr(sum(Ps)))"
    @assert isapprox(tr(sum(Psϵ)), N, atol=1e-7) "Not closed tr(sum(Psϵ))=$(tr(sum(Psϵ))) (en_basis)"
    if print==true
        println("TEST 1 PASSED!")
    end
    
    for i in eachindex(Ps)
        @assert isapprox(tr(Ps[i]), tr(Ps[i]*Ps[i]), atol=1e-7) "$(i)-th entry not a projector"
        @assert isapprox(tr(Psϵ[i]), tr(Psϵ[i]*Psϵ[i]), atol=1e-7) "$(i)-th entry not a projector (en_basis)"
    end
    if print==true
        println("TEST 2 PASSED!")
    end
    
    for i in eachindex(Ps)
        i == Pjs.number && break
        for j in i+1:Pjs.number
            @assert isapprox(tr(Ps[i]*Ps[j]), 0.0, atol=1e-7) "Projectors not orthogonal, i=$(i), j=$(j)"
            @assert isapprox(tr(Psϵ[i]*Psϵ[j]), 0.0, atol=1e-7) "Projectors not orthogonal (en_basis), i=$(i), j=$(j)"
        end
    end
    if print==true
        println("TEST 3 PASSED!")
    end
    
    for i in eachindex(Ps)
        evls = eigvals(Ps[i])
        evlsϵ = eigvals(Psϵ[i])
        for (evla, evlb) in zip(evls, evlsϵ)
            @assert (isapprox(evla, 0.0, atol=1e-7) || isapprox(evla, 1.0, atol=1e-7)) 
                    "Eigenvalue is not 0 or 1, evla = $(evla)" 
            @assert (isapprox(evlb, 0.0, atol=1e-7) || isapprox(evlb, 1.0, atol=1e-7)) 
                    "Eigenvalue is not 0 or 1, evlb = $(evlb)" 
        end
    end
    if print==true
        println("TEST 4 PASSED!")
        println("ALL PROJECTOR TESTS PASSED!")
    end
    
    return "ALL PROJECTOR TESTS PASSED!"
end


function test_dm(ρ::Array{ComplexF64}; print=true)
   
    @assert isapprox(tr(ρ), 1.0, atol=1e-7) "The trace of the density matirx is $(tr(ρ))"
    if print==true
        println("TEST 1 PASSED!")
    end
    
    evals = eigvals(ρ)
    
    for eval in evals
        @assert isapprox(imag(eval), 0.0, atol=1e-7) "eval=$(eval)"
    end
    if print==true
        println("TEST 2 PASSED!")
    end
    
    for eval in evals
        @assert real(eval)>=-1.0e-10 "eval=$(eval)"
    end
    if print==true
        println("TEST 3 PASSED!")
        println("ALL DENSITY MATRIX TESTS PASSED!")
    end
    
    return "ALL DENSITY MATRIX TESTS PASSED!"
end
