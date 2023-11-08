using PastaQ
using ITensors
using Random
using Printf
using OptimKit
using Zygote
using Dates

"""
This scripts demonstrates how to simulate quantum circuits using Julia and Tensor Networks.

It is based on the use of ITensors and PastaQ. This example is similar to the one discussed in 
the PastaQ github repo.
"""
N = 30   # number of qubits
J = -1.0  # Ising exchange interaction
h = 1.0  # transverse magnetic field

# We need to define a "hilbert space": a register for the qubits
hilbert = qubits(N)

# This creates a Hamiltonian that can be later expressed as an MPO
function ising_hamiltonian(N; J, h)
  os = OpSum()
  for j in 1:(N - 1)
    os += +J, "X", j, "X", j + 1
  end
  for j in 1:N
    os += +h, "Z", j
  end
  return os
end

# define the Hamiltonian
os = ising_hamiltonian(N; J, h)

# build MPO "cost function"
H = MPO(os, hilbert)

# Now we can calculate the GS of this Hamiltonian efficiently using DMRG
nsweeps = 10
maxdim = [10, 20, 30, 50, 100]
cutoff = 1e-10
Edmrg, Φ = dmrg(H, randomMPS(hilbert); outputlevel=0, nsweeps, maxdim, cutoff);
@printf("\nGround state energy from DMRG: %.10f\n\n", Edmrg)

# layer of single-qubit Ry gates
Rylayer(N, θ) = [("Ry", j, (θ=θ[j],)) for j in 1:N]

# brick-layer of CX gates
function CXlayer(N, Π)
  start = isodd(Π) ? 1 : 2
  return [("CX", (j, j + 1)) for j in start:2:(N - 1)]
end

# variational ansatz
function variationalcircuit(N, depth, θ)
  circuit = Tuple[]
  for d in 1:depth
    circuit = vcat(circuit, CXlayer(N, d))
    circuit = vcat(circuit, Rylayer(N, θ[d]))
  end
  return circuit
end

depth = 5
ψ = productstate(hilbert)

cutoff = 1e-8
maxdim = 100

# cost function
function loss(θ)
  circuit = variationalcircuit(N, depth, θ)
  Uψ = runcircuit(ψ, circuit; cutoff, maxdim)
  return inner(Uψ', H, Uψ; cutoff, maxdim)
end

Random.seed!(1234)

# initialize parameters
θ₀ = [2π .* rand(N) for _ in 1:depth]

start_time = Dates.now()
# run VQE using BFGS optimization
optimizer = LBFGS(; maxiter=50, verbosity=2)
function loss_and_grad(x)
  y, (∇,) = withgradient(loss, x)
  return y, ∇
end
θ, fs, gs, niter, normgradhistory = optimize(loss_and_grad, θ₀, optimizer)
@printf("Relative error: %.3E", abs(Edmrg - fs[end]) / abs(Edmrg))

end_time = Dates.now()
println("\n\n%%%%%%%% Total runtime was:\n$(Dates.canonicalize(end_time - start_time))")