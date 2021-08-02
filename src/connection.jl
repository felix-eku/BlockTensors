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
Base.get(default, d::SectorDims{S}, s::S) where S <: SymmetrySector = get(default, d.dims, s)
Base.get(d::SectorDims{S}, s::S, default) where S <: SymmetrySector = get(d.dims, s, default)

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

dual(a::Outgoing, b::Incoming) = a.space == b.space
dual(a::Incoming, b::Outgoing) = a.space == b.space
dual(::Connector, ::Connector) = false

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


Arrangement{N, S} = Dict{NTuple{N, S}, Tuple{S, UnitRange{Int}}}
function combine(
    connectors::NTuple{N, Connector}, dims::NTuple{N, SectorDims{S}}
) where {N, S <: SymmetrySector}
    combconnector = combine(connectors)
    CombinedDims = Vector{Pair{NTuple{N, S}, Int}}
    combs = DefaultDict{S, CombinedDims}(CombinedDims)
    for sector_dim_pairs in Iterators.product(dims)
        sectors, dims = zip(sector_dim_pairs)
        combsector = combine(combconnector, combine(connectors, sectors))
        push!(totals[combsector], sectors => prod(dims))
    end
    totaldims = Vector{S, Int}
    sizehint!(totaldims, length(combs))
    arrangement = Arrangement{N, S}()
    sizehint!(arrangement, sum(length, values(combs)))
    for (combsector, combdims) in combs
        totaldim = 0
        merge!(arrangement, (
                combsector, (totaldim + 1) : (totaldim += dim)
                for (sectors, dim) in combdims
            )
        )
        push!(totaldims, combsector => totaldim)
    end
    return combconnector, SectorDims(totaldims), arrangement
end


mutable struct UniqueToken end

struct Leg{S <: SymmetrySector, C <: Connector, N, Ls <: Tuple}
    token::UniqueToken
    connector::C
    dimensions::SectorDims{S}
    components::Ls
    arrangement::Arrangement{N, S}
    function Leg(
        connector::C, dimensions::SectorDims{S}
    ) where {S <: SymmetrySector, C <: Connector}
        new{S, C, 0, Tuple{}}(UniqueToken(), connector, dimensions, (), Arrangement{0, S}())
    end
    function Leg(changespace, legs::Tuple{Leg{S}, Vararg{Leg{S}}}) where {S <: SymmetrySector}
        connector, dimensions, arrangement = combine(
            getfield.(legs, :connector), getfield.(legs, :dimensions)
        )
        C = typeof(connector)
        connector = C(changespace(connector.space))
        new{S, C, length(legs), typeof(legs)}(
            UniqueToken(), connector, dimensions, legs, arrangement
        )
    end
    function Leg(
        leg::Leg{S, C, N, Ls}; dual::Bool = false, keep_token::Bool = false
    ) where {S <: SymmetrySector, C <: Connector, Ls <: Tuple}
        new{S, C, N, Ls}(
            keep_token ? leg.token : UniqueToken(),
            dual ? dual(leg.connector) : leg.connector,
            leg.dimensions, leg.components, leg.arrangement
        )
    end
end
function Leg(legs::Tuple{Leg{S}, Vararg{Leg{S}}}) where S <: SymmetrySector
    Leg(identity, legs)
end
Leg(legs::Leg...) = Leg(identity, legs)
function Leg(space::Space, legs::Tuple{Leg{S}, Vararg{Leg{S}}}) where S <: SymmetrySector
    Leg(_ -> space, legs)
end
Leg(space::Space, legs::Leg...) = Leg(space, legs)
Leg(changespace, legs::Leg...) = Leg(changespace, legs)

function Leg(leg::Leg{S}, dimensions::SectorDims{S}) where S <: SymmetrySector
    leg.dimensions == dimensions ||
        throw(ArgumentError("Leg has different dimensions than specified"))
    return leg
end

function Base.:(==)(a::Leg, b::Leg)
    a.connector == b.connector &&
        a.dimensions == b.dimensions && a.components == b.components || return false
    @assert a.arrangement == b.arrangement

end

function Base.hash(leg::Leg, h::UInt)
    h = hash(leg.connector, h)
    h = hash(leg.dimensions, h)
    h = hash(leg.components, h)
    return h
end

function matching(a::Leg, b::Leg)
    a.token == b.token && a.connector == b.connector || return false
    @assert a.dimensions == b.dimensions && 
        a.components == b.components && a.arrangement == b.arrangement
    return true
end
matching(x::Union{Space, Connector}, match::Leg) = matching(x, match.connector)

dual(leg::Leg) = Leg(leg, dual = true, keep_token = true)
function dual(a::Leg, b::Leg)
    a.token == b.token && a.connector == dual(b.connector) || return false
    @assert a.dimensions == b.dimensions && 
        a.components == b.components && a.arrangement == b.arrangement
    return true
end

function Base.show(io::IO, leg::Leg{S, C, 0}) where {S <: SymmetrySector, C <: Connector}
    show(io, Leg)
    print(io, "(")
    show(io, leg.connector)
    print(io, ", ")
    show(io, leg.dimensions)
    print(io, ")")
end

function Base.show(io::IO, leg::Leg)
    show(io, Leg)
    print(io, "(")
    show(io, leg.connector.space)
    print(io, ", ")
    show(io, leg.legs)
    print(io, ")")
end

Base.show(io::IO, mime::MIME"text/plain", leg::Leg) = show(io, mime, leg.connector)


function invertarrangement(arrangement::Arrangement{S, N}) where {S <: SymmetrySector, N}
    RangeSectorVector = Vector{Pair{UnitRange{Int}, NTuple{N, S}}}
    rangesectors = DefaultDict{S, RangeSectorVector}(RangeSectorVector)
    for (sectors, (sector, range)) in arrangement
        push!(rangesectors[sector], range => sectors)
    end
    return rangesectors
end


function matchingpermutations(a::Tuple, b::Tuple; matchs = matching)
    perm_a = collect(only(axes(a)))
    perm_b = collect(only(axes(b)))
    m = 1
    @assert m == firstindex(perm_a) == firstindex(perm_b) "array indices don't start at 1 ?!?"
    for (i_a, k_a) in enumerate(perm_a)
        c_a = a[k_a]
        unmatched_b = @view perm_b[m:end]
        for (i_b, k_b) in enumerate(unmatched_b)
            c_b = b[k_b]
            if matchs(c_b, c_a)
                if i_a > m
                    perm_a[m], perm_a[i_a] = perm_a[i_a], perm_a[m]
                end
                # index 1 in unmatched_b corresponds to index m in inds_b
                if i_b > 1
                    unmatched_b[1], unmatched_b[i_b] = unmatched_b[i_b], unmatched_b[1]
                end
                m += 1
                break
            end
        end
    end
    return perm_a, perm_b, m - 1
end
