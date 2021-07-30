using DataStructures

struct SectorDims{S <: SymmetrySector}
    dims::OrderedDict{S, Int}
    totaldim::Int
    function SectorDims(dimensions::AbstractVector{Pair{S, Int}}) where S <: SymmetrySector
        sort!(dimensions, by = first)
        allunique_sorted(d.first for d in dimensions) || throw(
            ArgumentError("multiple dimensions for the same SymmetrySector")
        )
        all(dimension -> dimension.second > 0, dimensions) || throw(
            ArgumentError("negative dimension")
        )
        new{S}(OrderedDict{S, Int}(dimensions), sum(getfield.(dimensions, :second)))
    end
end
function SectorDims(dimensions)
    P = eltype(dimensions)
    P <: Pair || throw(ArgumentError("dimensions should be a collection of Pairs"))
    S = P.parameters[1]
    SectorDims(collect(Pair{S, Int}, dimensions))
end
SectorDims(dimensions::Pair...) = SectorDims(dimensions)

Base.getindex(d::SectorDims{S}, s::S) where S <: SymmetrySector = d.dims[s]

Base.iterate(d::SectorDims) = iterate(d.dims)
Base.iterate(d::SectorDims, state) = iterate(d.dims, state)

function Base.:(==)(a::SectorDims{S}, b::SectorDims{S}) where S <: SymmetrySector
    a.dims == b.dims || return false
    @assert a.totaldim == b.totaldim
    return true
end

function Base.hash(d::SectorDims, h::UInt)
    h = hash(typeof(d), h)
    for r in d.dims
        h = hash(r, h)
    end
    return h
end

function Base.show(io::IO, d::SectorDims)
    T = typeof(d)
    show(io, ifelse((:typeinfo => T) in io || !isempty(d.dims), SectorDims, T))
    print(io, "(")
    join(io, d.dims, ", ")
    print(io, ")")
end


SymbolOrString = Union{Symbol, AbstractString}

struct Space{S <: SymmetrySector}
    name::Symbol
    tags::Dict{Symbol, Any}
end
function Space{S}(name::SymbolOrString; tags...) where S <: SymmetrySector
    Space{S}(Symbol(name), Dict(pairs(tags)))
end

function Base.getproperty(space::Space, name::Symbol)
    if name ≡ :name
        getfield(space, :name)
    elseif name ≡ :tags
        getfield(space, :tags)
    else
        getfield(space, :tags)[name]
    end
end
function Base.propertynames(space::Space, private::Bool = false)
    tags = keys(space.tags)
    private ? (:name, tags..., :tags) : (:name, tags...)
end

function Base.:(==)(a::Space{S}, b::Space{S}) where S <: SymmetrySector
    a.name == b.name && a.tags == b.tags
end

function Base.hash(x::Space, h::UInt)
    h = hash(typeof(x), h)
    h = hash(x.name, h)
    for tag in getfield(x, :tags)
        h ⊻= hash(tag)
    end
    return h
end

function Base.show(io::IO, space::Space)
    T = typeof(space)
    show(io, ifelse((:typeinfo => T) in io, Space, T))
    show(io, MIME"text/no-type"(), space)
end

function Base.show(io::IO, ::MIME"text/no-type", space::Space)
    compact = get(io, :compact, false)
    print(io, "(")
    show(io, space.name)
    for (label, value) in space.tags
        print(io, ", ", label, compact ? "=" : " = ")
        show(io, value)
    end
    print(io, ")")
end

function Base.show(io::IO, ::MIME"text/plain", space::Space)
    sep = get(io, :compact, false) ? "" : " "
    print(io, space.name)
    for (label, value) in space.tags
        print(io, sep, "|", sep, label, sep, "=", sep)
        show(io, value)
    end
end

function combine(spaces::Tuple{Space{S}, Vararg{Space{S}}}) where S <: SymmetrySector
    Space{S}(
        Symbol(unique(getfield.(spaces, :name))...),
        mergewith(union, getfield.(spaces, :tags)...)
    )
end
combine(spaces::Space{S}...) where S <: SymmetrySector = combine(spaces)

matching(x, match) = false
matching(x) = Base.Fix1(matching, x)
matching(x, matches::Tuple) = filter(matching(x), matches)

function matching(x::Space{S}, match::Space{S}) where S <: SymmetrySector
    x.name == match.name && all(x.tags) do (label, values)
        haskey(match.tags, label) && values ⊆ match.tags[label]
    end
end


abstract type Connector{S <: SymmetrySector} end

for Connect = (:Incoming, :Outgoing)
    eval(quote

    struct $Connect{S <: SymmetrySector} <: Connector{S}
        space::Space{S}
    end
    function $Connect{S}(name::SymbolOrString; tags...) where S <: SymmetrySector
        $Connect{S}(Space{S}(name; tags...))
    end
    $Connect(c::Connector{S}) where S <: SymmetrySector = $Connect{S}(c.space)

    Base.:(==)(a::$Connect, b::$Connect) = a.space == b.space

    function Base.show(io::IO, c::$Connect)
        T = typeof(c)
        show(io, ifelse((:typeinfo => T) in io, $Connect, T))
        show(io, MIME"text/no-type"(), c.space)
    end

    function matching(x::$Connect{S}, match::$Connect{S}) where S <: SymmetrySector
        matching(x.space, match.space)
    end
    
    end)
end

Base.hash(x::Connector, h::UInt) = hash(x.space, hash(typeof(x), h))

matching(x::Space, match::Connector) = matching(x, match.space)

function Base.show(io::IO, mime::MIME"text/plain", c::Connector)
    sep = get(io, :compact, false) ? "" : " "
    show(io, mime, c.space)
    print(io, sep, "|", sep, direction_label(c))
end
direction_label(::Incoming) = "in"
direction_label(::Outgoing) = "out"

dual(c::Incoming) = Outgoing(c)
dual(c::Outgoing) = Incoming(c)

function combine(
    connectors::Tuple{Connector{S}, Vararg{Connector{S}}}
) where S <: SymmetrySector
    total_direction(connectors)(combine(getfield.(connectors, :space)))
end
combine(connectors::Connector{S}...) where S <: SymmetrySector = combine(connectors)

function total_direction(
    connectors::Tuple{Connector{S}, Vararg{Connector{S}}}
) where S <: SymmetrySector
    balance = sum(Int, connectors, init = 0)
    balance == 0 ? typeof(first(connectors)) : Connector{S}(balance)
end
total_direction(connectors::Connector...) = total_direction(connectors)

(::Type{T})(::Incoming) where {T <: Real} = -one(T)
(::Type{T})(::Outgoing) where {T <: Real} = +one(T)

Connector{S}(x::Real) where S <: SymmetrySector = x < 0 ? Incoming{S} : Outgoing{S}

combine(::Incoming{S}, sector::S) where S <: SymmetrySector = -sector
combine(::Outgoing{S}, sector::S) where S <: SymmetrySector = sector

function combine(
    connectors::NTuple{N, Connector{S}}, sectors::NTuple{N, S}
) where {N, S <: SymmetrySector}
    mapreduce(combine, +, connectors, sectors, init=zero(S))
end


mutable struct UniqueToken end

struct Leg{S <: SymmetrySector, C <: Connector{S}, Ls <: Tuple}
    token::UniqueToken
    connector::C
    dimensions::SectorDims{S}
    components::Ls
    function Leg(
        connector::C, dimensions::SectorDims{S}
    ) where {S <: SymmetrySector, C <: Connector{S}}
        new{S, C, Tuple{}}(UniqueToken(), connector, dimensions, ())
    end
    function Leg(
        connector::C, legs::Tuple{Leg{S}, Vararg{Leg{S}}}
    ) where {S <: SymmetrySector, C <: Connector{S}}
        dimensions = combine(getfield.(legs, :connector), getfield.(legs, :dimensions))
        new{S, C, typeof(legs)}(UniqueToken(), connector, dimensions, legs)
    end
    function Leg(
        leg::Leg{S, C, Ls}; dual::Bool = false
    ) where {S <: SymmetrySector, C <: Connector{S}, Ls <: Tuple}
        new{S, C, Ls}(
            leg.token, 
            dual ? dual(leg.connector) : leg.connector,
            leg.dimensions, leg.components
        )
    end
end

Base.hash(leg::Leg, h::UInt) = hash(leg.token, hash(typeof(leg), h))

function Base.:(==)(a::Leg, b::Leg)
    a.token == b.token && a.connector == b.connector || return false
    @assert a.dimensions == b.dimensions && a.components == b.components
    return true
end

function matching(a::Leg, b::Leg)
    a.token == b.token && a.connector == dual(b.connector) || return false
    @assert a.dimensions == b.dimensions && a.components == b.components
    return true
end

matching(x::Union{Space, Connector}, match::Leg) = matching(x, match.connector)

dual(leg::Leg) = Leg(leg, dual = true)
