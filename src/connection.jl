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

function compatible(a::SectorDims{S}, b::SectorDims{S}) where S <: SymmetrySector
    all(pairs(a.dims)) do (sector, dim_a)
        dim_b = get(b.dims, sector, nothing)
        dim_b ≡ nothing || dim_a == dim_b
    end
end
compatible(a::SectorDims, b::SectorDims) = false

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
function Space(space::Space; extratags...)
    Space(space.name, mergewith(union, space.tags, pairs(extratags)))
end

addtags!(space::Space; tags...) = mergewith!(union, space.tags, pairs(tags))

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

Base.iterate(space::Space, state = ()) = state ≡ nothing ? nothing : (space, nothing)
Base.broadcastable(space::Space) = Ref(space)

Base.copy(a::Space) = Space(a.name, copy(a.tags))

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

    Connector(c::$Connect; extratags...) = $Connect(Space(c.space; extratags...))

    Base.:(==)(a::$Connect, b::$Connect) = a.space == b.space

    Base.copy(a::$Connect) = $Connect(copy(a.space))

    matching(c::$Connect, match::$Connect) = matching(c.space, match.space)

    end)
end

Base.hash(c::Connector, h::UInt) = hash(c.space, hash(typeof(c), h))

Base.iterate(c::Connector, state = ()) = state ≡ nothing ? nothing : (c, nothing)
Base.broadcastable(c::Connector) = Ref(c)

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

addtags!(c::Connector; tags...) = addtags!(c.space; tags...)


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
    for sector_dim_pairs in Iterators.product(dims...)
        sectors, dims = zip(sector_dim_pairs...)
        combsector = combine(combconnector, combine(connectors, sectors))
        push!(combs[combsector], sectors => prod(dims))
    end
    totaldims = Vector{Pair{S, Int}}()
    sizehint!(totaldims, length(combs))
    arrangement = Arrangement{N, S}()
    sizehint!(arrangement, sum(length, values(combs)))
    for (combsector, combdims) in combs
        totaldim = 0
        for (sectors, dim) in combdims
            arrangement[sectors] = combsector, (totaldim + 1) : (totaldim += dim)
        end
        push!(totaldims, combsector => totaldim)
    end
    return combconnector, SectorDims(totaldims), arrangement
end


module ConnectionIds

export ConnectionId, nextid

maxid = UInt(0)

mutable struct ConnectionId
    id::UInt
    function ConnectionId(id::UInt)
        global maxid = max(maxid::UInt, id)
        new(id)
    end
end
ConnectionId(id::Integer) = ConnectionId(convert(UInt, id))
ConnectionId() = ConnectionId(nextid())

function Base.setproperty!(x::ConnectionId, name::Symbol, id::UInt)
    name ≡ :id || throw(ErrorException("type ConnectionId has no field $name"))
    global maxid = max(maxid::UInt, id)
    setfield!(x, name, id)
end
function Base.setproperty!(x::ConnectionId, name::Symbol, id::Integer)
    setproperty!(x, name, convert(UInt, id))
end

Base.copy(x::ConnectionId) = ConnectionId(x.id)

Base.:(==)(a::ConnectionId, b::ConnectionId) = a.id == b.id

nextid() = maxid::UInt + 1

end


using .ConnectionIds

struct Leg{S <: SymmetrySector, C <: Connector, Ls <: Tuple, N}
    connection::ConnectionId
    connector::C
    dimensions::SectorDims{S}
    components::Ls
    arrangement::Arrangement{N, S}
    function Leg(
        connection::ConnectionId, connector::C, dimensions::SectorDims{S}, 
        components::NTuple{N, Leg{S}}, arrangement::Arrangement{N, S}
    ) where {S <: SymmetrySector, C <: Connector, N}
        new{S, C, typeof(components), N}(
            connection, connector, dimensions, components, arrangement
        )
    end
end
function Leg(connector::Connector, dimensions::SectorDims{S}) where S <: SymmetrySector
    Leg(ConnectionId(), connector, dimensions, (), Arrangement{0, S}())
end
function Leg(changespace, legs::Tuple{Leg{S}, Vararg{Leg{S}}}) where S <: SymmetrySector
    connector, dimensions, arrangement = combine(
        getfield.(legs, :connector), getfield.(legs, :dimensions)
    )
    connector = typeof(connector)(changespace(connector.space))
    Leg(ConnectionId(0), connector, dimensions, legs, arrangement)
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
    dimensions.dims ⊆ leg.dimensions.dims ||
        throw(ArgumentError("Leg has different dimensions than specified"))
    return Leg(leg.connection, leg.connector, dimensions, leg.components, leg.arrangement)
end

function Base.copy(leg::Leg)
    Leg(
        copy(leg.connection), copy(leg.connector), 
        leg.dimensions, leg.components, leg.arrangement
    )
end

addtags!(leg::Leg; tags...) = addtags!(leg.connector.space; tags...)

function connect!(a::Leg, b::Leg)
    @assert dual(a.connector, b.connector) && compatible(a.dimensions, b.dimensions)
    id = nextid()
    a.connection.id = b.connection.id = id
end


function Base.hash(leg::Leg, h::UInt)
    h = hash(leg.connector, h)
    h = hash(leg.dimensions, h)
    h = hash(leg.components, h)
    return h
end

function Base.:(==)(a::Leg, b::Leg)
    a.connector == b.connector &&
        a.dimensions == b.dimensions && a.components == b.components || return false
    @assert a.arrangement == b.arrangement
    return true
end

matching(a::Leg, b::Leg) = a.connector == b.connector
matching(x::Union{Space, Connector}, match::Leg) = matching(x, match.connector)

dual(leg::Leg; connect::Bool = false) = Leg(
    connect ? copy(leg.connection) : ConnectionId(),
    dual(leg.connector), leg.dimensions, 
    dual.(leg.components), leg.arrangement
)
function dual(a::Leg, b::Leg)
    dual(a.connector, b.connector) || return false
    @assert a.dimensions == b.dimensions
    return true
end

function connected(a::Leg, b::Leg)
    a.connection == b.connection && dual(a.connector, b.connector) || return false
    if a.connection.id == 0
        length(a.components) == length(b.components) || return false
        all(connected.(a.components, b.components)) || return false
        @assert a.arrangement == b.arrangement
        connect!(a, b)
    end
    compatible(a.dimensions, b.dimensions)
    return true
end


function Base.show(
    io::IO, leg::Leg{S, C, Tuple{}, 0}
) where {S <: SymmetrySector, C <: Connector}
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
    show(io, leg.components)
    print(io, ")")
end

Base.show(io::IO, mime::MIME"text/plain", leg::Leg) = show(io, mime, leg.connector)


function invertarrangement(arrangement::Arrangement{N, S}) where {S <: SymmetrySector, N}
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
