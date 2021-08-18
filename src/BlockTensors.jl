module BlockTensors

using Base.Iterators
using LinearAlgebra
using DataStructures

export SymmetrySector, Trivial, @SymmetrySector
export Space, Connector, Incoming, Outgoing, SectorDims, Leg, LegLike
export dual, matching, connected, connect!, combine, addtags!
export Tensor, mergelegs, separatelegs, svd, qr, lq

include("util.jl")
include("symmetry.jl")
include("connection.jl")
include("tensor.jl")

include("TensorChain.jl")
include("MatrixProductStates.jl")

end
