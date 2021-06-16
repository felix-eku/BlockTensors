module Tensors

import Base.ImmutableDict

abstract type Sector end

# TODO: macro to create Sector types

struct Space{S <: Sector}
    dimensions::ImmutableDict{S, Int}
    dual::Bool
end

struct Connection{S <: Sector}
    name::Symbol
    space::Space{S}
    tags::ImmutableDict{Symbol, Int}
end

struct Tensor{T <: Number, S <: Sector, N} 
    components::Dict{NTuple{N, S}, Array{T, N}}
    connections::NTuple{N, Connection{S}}
end

end