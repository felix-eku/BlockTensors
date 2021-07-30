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

struct Space
    name::Symbol
    tags::Dict{Symbol, Any}
end
Space(name::SymbolOrString; tags...) = Space(Symbol(name), Dict(pairs(tags)))

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

Base.:(==)(a::Space, b::Space) = a.name == b.name && a.tags == b.tags

function Base.hash(space::Space, h::UInt)
    h = hash(Space, h)
    h = hash(space.name, h)
    for tag in space.tags
        h ⊻= hash(tag)
    end
    return h
end

function Base.show(io::IO, space::Space)
    show(io, Space)
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

function combine(spaces::Tuple{Space, Vararg{Space}})
    Space(
        Symbol(unique(getfield.(spaces, :name))...),
        mergewith(union, getfield.(spaces, :tags)...)
    )
end
combine(spaces::Space...) = combine(spaces)

matching(x, match) = false
matching(x) = Base.Fix1(matching, x)
matching(x, matches::Tuple) = filter(matching(x), matches)

function matching(space::Space, match::Space)
    space.name == match.name && all(space.tags) do (label, values)
        haskey(match.tags, label) && values ⊆ match.tags[label]
    end
end


abstract type Connector end

for Connect = (:Incoming, :Outgoing)
    eval(quote

    struct $Connect <: Connector
        space::Space
    end
    $Connect(name::SymbolOrString; tags...) = $Connect(Space(name; tags...))
    $Connect(c::Connector) = $Connect(c.space)

    Base.:(==)(a::$Connect, b::$Connect) = a.space == b.space

    matching(c::$Connect, match::$Connect) = matching(c.space, match.space)
    
    end)
end

Base.hash(c::Connector, h::UInt) = hash(c.space, hash(typeof(c), h))

matching(space::Space, match::Connector) = matching(space, match.space)

function Base.show(io::IO, c::Connector)
    show(io, typeof(c))
    show(io, MIME"text/no-type"(), c.space)
end

function Base.show(io::IO, mime::MIME"text/plain", c::Connector)
    sep = get(io, :compact, false) ? "" : " "
    show(io, mime, c.space)
    print(io, sep, "|", sep, direction_label(c))
end
direction_label(::Incoming) = "in"
direction_label(::Outgoing) = "out"

dual(c::Incoming) = Outgoing(c)
dual(c::Outgoing) = Incoming(c)

function combine(connectors::Tuple{Connector, Vararg{Connector}})
    total_direction(connectors)(combine(getfield.(connectors, :space)))
end
combine(connectors::Connector...) = combine(connectors)

function total_direction(connectors::Tuple{Connector, Vararg{Connector}})
    balance = sum(Int, connectors, init = 0)
    balance == 0 ? typeof(first(connectors)) : Connector(balance)
end
total_direction(connectors::Connector...) = total_direction(connectors)

(::Type{T})(::Incoming) where {T <: Real} = -one(T)
(::Type{T})(::Outgoing) where {T <: Real} = +one(T)

Connector(x::Real) = x < 0 ? Incoming : Outgoing


combine(::Incoming, sector::SymmetrySector) = -sector
combine(::Outgoing, sector::SymmetrySector) = sector

function combine(
    connectors::NTuple{N, Connector}, sectors::NTuple{N, S}
) where {N, S <: SymmetrySector}
    mapreduce(combine, +, connectors, sectors, init=zero(S))
end


mutable struct UniqueToken end

struct Leg{S <: SymmetrySector, C <: Connector, Ls <: Tuple}
    token::UniqueToken
    connector::C
    dimensions::SectorDims{S}
    components::Ls
    function Leg(
        connector::C, dimensions::SectorDims{S}
    ) where {S <: SymmetrySector, C <: Connector}
        new{S, C, Tuple{}}(UniqueToken(), connector, dimensions, ())
    end
    function Leg(
        connector::C, legs::Tuple{Leg{S}, Vararg{Leg{S}}}
    ) where {S <: SymmetrySector, C <: Connector}
        dimensions = combine(getfield.(legs, :connector), getfield.(legs, :dimensions))
        new{S, C, typeof(legs)}(UniqueToken(), connector, dimensions, legs)
    end
    function Leg(
        leg::Leg{S, C, Ls}; dual::Bool = false
    ) where {S <: SymmetrySector, C <: Connector, Ls <: Tuple}
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
