module Tensors

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

function Base.:+(x::S, ys::S...) where S <: SymmetrySector
    S(; 
        (
            field => +(getfield(x, field), map(y -> getfield(y, field), ys)...) 
            for field in fieldnames(S)
        )...
    )
end

function Base.:-(x::S) where S <: SymmetrySector
    S(; (field => -(getfield(x, field)) for field in fieldnames(S))...)
end

function Base.show(io::IO, x::S) where S <: SymmetrySector
    show(io, S)
    print(io, "(")
    sep = ""
    for field in fieldnames(S)
        print(io, sep, field, " = ")
        show(io, getfield(x, field))
        sep = ", "
    end
    print(io, ")")
end

struct Connection{S <: SymmetrySector}
    dims::Dict{S, Int}
    name::Symbol
    tags::Dict{Symbol, Any}
end
function Connection(
    dims::AbstractDict{S, <:Integer}, 
    name::Union{Symbol, AbstractString}, 
    tags::Pair{Symbol}...
) where S <: SymmetrySector
    Connection{S}(convert(Dict{S, Int}, dims), Symbol(name), Dict{Symbol, Any}(tags...))
end
Connection(dim::Integer, name, tags...) = Connection(Dict(Trivial() => dim), name, tags...)

struct Connector{S <: SymmetrySector}
    connection::Connection{S}
    out::Bool
end

struct Tensor{T <: Number, S <: SymmetrySector, N} 
    components::Dict{NTuple{N, S}, Array{T, N}}
    connectors::NTuple{N, Connector{S}}
end

end