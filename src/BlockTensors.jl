module BlockTensors

using Base.Iterators
using LinearAlgebra
using DataStructures

export SymmetrySector, Trivial, @SymmetrySector
export Space, Connector, Incoming, Outgoing, SectorDims, Leg, LegLike
export dual, matching, matchingpermutations, combine
export Tensor, mergelegs, separatelegs, svd, qr, lq

include("util.jl")
include("symmetry.jl")
include("connection.jl")
include("tensor.jl")

end
