module Tensors

import Base.ImmutableDict

abstract type SymmetrySector end

struct Trivial <: SymmetrySector end

macro SymmetrySector(name::Symbol, numbers)
    # adapted from @NamedTuple
    name = esc(name)
    Meta.isexpr(numbers, [:braces, :block]) || throw(
        ArgumentError("@Sector expects {...} or begin ... end as second argument")
    )
    decls = filter(e -> !(e isa LineNumberNode), numbers.args)
    all(
        e -> e isa Symbol || 
            (Meta.isexpr(e, :(::)) && e.args[1] isa Symbol && eval(e.args[2]) <: Integer),
        decls
    ) || throw(
        ArgumentError(
            "second argument of @Sector must be a sequence of name[::type] expressions"
            * " with type <: Integer"
        )
    )
    fields = Symbol[e isa Symbol ? e : e.args[1] for e in decls]
    types = [esc(e isa Symbol ? Int : e.args[2]) for e in decls]
    numbers = (:($field::$type) for (field, type) in zip(fields, types))
    args = (Expr(:kw, :($(esc(field))::Integer), 0) for field in fields)
    conversions = (
        :(convert($type, $(esc(field)))) for (field, type) in zip(fields, types)
    )
    constructor = :($name(; $(args...)) = new($(conversions...)))
    return Expr(
        :struct, false,                         # declare an immutable struct
        :($name <: SymmetrySector),
        Expr(:block, numbers..., constructor)
    )
end

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