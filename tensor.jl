module Tensors

import Base.ImmutableDict

abstract type SymmetrySector end

# TODO: macro to create Sector types

struct Space{S <: SymmetrySector}
    dimensions::ImmutableDict{S, Int}
    dual::Bool
end

struct Connection{S <: SymmetrySector}
    name::Symbol
    space::Space{S}
    tags::ImmutableDict{Symbol, Int}
end

struct Tensor{T <: Number, S <: SymmetrySector, N} 
    components::Dict{NTuple{N, S}, Array{T, N}}
    connections::NTuple{N, Connection{S}}
end

end