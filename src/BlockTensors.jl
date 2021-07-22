module BlockTensors

using Base.Iterators
using LinearAlgebra
using DataStructures

export SymmetrySector, Trivial, @SymmetrySector
export Connection, Connector, replicate, dual, matchconnectors
export Tensor, mergeconnectors, separateconnectors, svd, qr, lq

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

function Base.isless(x::S, y::S) where S <: SymmetrySector
    for field in fieldnames(S)
        getfield(x, field) < getfield(y, field) || return false
    end
    return true
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


SymbolOrString = Union{Symbol, AbstractString}

struct Connection{S <: SymmetrySector}
    dims::Dict{S, Int}
    name::Symbol
    tags::Dict{Symbol, Any}
end
function Connection{S}(
    dims::AbstractDict, name::SymbolOrString; tags...
) where S <: SymmetrySector
    Connection{S}(convert(Dict{S, Int}, dims), Symbol(name), Dict(tags))
end
function Connection(
    dims::AbstractDict{S, <: Integer}, name::SymbolOrString; tags...
) where S <: SymmetrySector
    Connection{S}(dims, name; tags...)
end
function Connection{S}(name::SymbolOrString; tags...) where S <: SymmetrySector
    Connection(Dict{S, Int}(), name; tags...)
end
Connection(name::SymbolOrString; tags...) = Connection{Trivial}(name; tags...)
Connection{Trivial}(dim::Integer, name; tags...) = Connection(Dict(Trivial() => dim), name; tags...)
Connection(dim::Integer, name; tags...) = Connection{Trivial}(dim, name; tags...)

function replicate(c::Connection{S}) where S <: SymmetrySector
    Connection{S}(c.name; c.tags...)
end

function Base.show(io::IO, con::Connection{S}) where S <: SymmetrySector
    T = typeof(con)
    show(io, (:typeinfo => T) in io ? Connection : T)
    print(io, "(")
    show(IOContext(io, :typeinfo => Dict{S, Int}), con.dims)
    print(io, ", ")
    show(io, con.name)
    for (label, value) in con.tags
        print(io, ", ", label, get(io, :compact, false) ? "=" : " = ")
        show(io, value)
    end
    print(io, ")")
end

struct Connector{S <: SymmetrySector}
    connection::Connection{S}
    out::Bool
end
Connector(connection::Connection; out::Bool = false) = Connector(connection, out)

function replicate(c::Connector; dual::Bool = false)
    Connector(replicate(c.connection), dual ? !c.out : c.out)
end

dual(c::Connector) = Connector(c.connection, !c.out)

function Base.show(io::IO, con::Connector{S}) where S <: SymmetrySector
    T = typeof(con)
    show(io, (:typeinfo => T) in io ? Connector : T)
    print(io, "(")
    show(IOContext(io, :typeinfo => Connection{S}), con.connection)
    print(io, ", ", con.out)
    print(io, ")")
end

function matchconnectors(a::Tuple{Vararg{Connector}}, b::Tuple{Vararg{Connector}}; direct::Bool = false)
    @assert allunique(a) "Connectors a are not unique."
    @assert allunique(b) "Connectors b are not unique."
    inds_a = collect(only(axes(a)))
    inds_b = collect(only(axes(b)))
    m = 1
    @assert m == firstindex(inds_a) == firstindex(inds_b) "Array indices don't start at 1 ?!?"
    for (i_a, k_a) in enumerate(inds_a)
        con_a = a[k_a]
        unmatched_b = @view inds_b[m:end]
        for (i_b, k_b) in enumerate(unmatched_b)
            con_b = b[k_b]
            if direct ? con_a == con_b : con_a.connection ≡ con_b.connection
                @assert direct || (con_a.out ⊻ con_b.out) "Connectors for same connection do not match."
                if i_a > m
                    inds_a[m], inds_a[i_a] = inds_a[i_a], inds_a[m]
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
    return inds_a, inds_b, m - 1
end

function mergesectors(
    sectors::NTuple{N, S}, connectors::NTuple{N, Connector{S}}
) where {N, S <: SymmetrySector}
    mapreduce((s, c) -> c.out ? s : -s, +, sectors, connectors)
end

struct Tensor{T <: Number, S <: SymmetrySector, N} 
    components::Dict{NTuple{N, S}, Array{T, N}}
    connectors::NTuple{N, Connector{S}}
    function Tensor{T, S, N}(
        components::Dict{NTuple{N, S}, Array{T, N}}, 
        connectors::NTuple{N, Connector{S}};
        check::Bool = true
    ) where {T <: Number, S <: SymmetrySector, N}
        if check
            for (sectors, block) in components
                _checkblocksize(connectors, size(block), sectors)
            end
        end
        new(components, connectors)
    end
end
function Tensor{T}(t::Tensor{Tprime, S, N}) where {T <: Number, Tprime <: Number, S <: SymmetrySector, N}
    Tensor{T, S, N}(
        Dict(sector => Array{T, N}(block) for (sector, block) in t.components), 
        t.connectors, check = false
    )
end
function Tensor{T}(
    components::AbstractDict{NTuple{N, S}, <:AbstractArray{Tprime, N}},
    connectors::Vararg{Connector{S}, N}
) where {T <: Number, Tprime <: Number, S <: SymmetrySector, N}
    Tensor{T, S, N}(convert(Dict{NTuple{N, S}, Array{T, N}}, components), connectors)
end
function Tensor{T}(connectors::Vararg{Connector{S}, N}) where {T <: Number, S <: SymmetrySector, N}
    Tensor{T, S, N}(Dict{NTuple{N, S}, Array{T, N}}(), connectors, check = false)
end
function Tensor(
    components::AbstractDict{NTuple{N, S}, <:AbstractArray{T, N}}, 
    connectors::Vararg{Connector{S}, N}
) where {T <: Number, S <: SymmetrySector, N}
    Tensor{T}(components, connectors...)
end
function Tensor(
    components::AbstractArray{T, N}, 
    connectors::Vararg{Connector{Trivial}, N}
) where {T <: Number, N}
    sectors = ntuple(k -> Trivial(), Val(N))
    Tensor{T, Trivial, N}(Dict(sectors => components), connectors)
end

function _checkblocksize(
    connectors::NTuple{N, Connector{S}}, blocksize::NTuple{N, Int}, sectors::NTuple{N, S}
) where {S <: SymmetrySector, N}
    for k = 1:N
        dim = blocksize[k]
        dim == get!(connectors[k].connection.dims, sectors[k], dim) ||
            throw(
                DimensionMismatch(
                    "dimension $k of block for sectors $sectors does not match " *
                    "dimension specified by connection"
                )
            )
    end
end

rank(::Tensor{T, S, N}) where {T <: Number, S <: SymmetrySector, N} = N

function _connectorpermutation(
    t::Tensor{T, S, N}, connects::NTuple{N, Connector}
) where {T <: Number, S <: SymmetrySector, N}
    conrange, perm, n = matchconnectors(t.connectors, connects, direct = true)
    n == N || throw(ArgumentError("$connects is not a permutation of the connectors of the tensor"))
    @assert conrange == 1:N
    return perm
end

function _getsectors_indices(::Type{Trivial}, blockinds::NTuple{N}) where N
    sectors = ntuple(k -> Trivial(), Val(N))
    return sectors, blockinds
end
function _getsectors_indices(
    ::Type{S}, blockinds::Tuple{Vararg{Pair{S, <:Any}}}
) where S <: SymmetrySector
    sectors = getproperty.(blockinds, :first)
    inds = getproperty.(blockinds, :second)
    return sectors, inds
end
function _getsectors_indices(
    ::Type{S}, sectors::Tuple{Vararg{S}}
) where S <: SymmetrySector
    inds = fill(:, length(sectors))
    return sectors, inds
end

function Base.getindex(
    t::Tensor{T, S, N}, 
    indices::Vararg{Pair{Connector{S}, <: Any}, N}
) where {T <: Number, S <: SymmetrySector, N}
    connects = getproperty.(indices, :first)
    perm = _connectorpermutation(t, connects)
    blockinds = getproperty.(indices, :second)
    sectors, inds = _getsectors_indices(S, blockinds)
    return PermutedDimsArray(t.components[sectors[perm]], invperm(perm))[inds...]
end

function Base.setindex!(
    t::Tensor{T, S, N},
    comps::Array{T, N},
    indices::Vararg{Pair{Connector{S}, S}}
) where {T <: Number, S <: SymmetrySector, N}
    connects = getproperty.(indices, :first)
    perm = _connectorpermutation(t, connects)
    sectors = getproperty.(indices, :second)[perm]
    sectors ∉ t.components || throw(ArgumentError("block for sectors $sectors already set"))
    block = permutedims(comps, perm)
    _checkblocksize(t.connectors, size(block), sectors)
    t.components[sectors] = block
end


function _map(
    f_binary, f_unary, a::Tensor{Ta, S, N}, bs::Vararg{Tensor{Tb, S, N} where Tb <: Number}
) where {Ta <: Number, S <: SymmetrySector, N}
    T = promote_type(Ta, map(TensorT -> TensorT.parameters[1], typeof(bs).parameters)...)
    t = Tensor{T}(a)
    for b in bs
        perm = _connectorpermutation(t, b.connectors)
        for (sectors_perm, block_b) in b.components
            sectors = sectors_perm[perm]
            block_b = PermutedDimsArray(block_b, perm)
            block = get(t.components, sectors, nothing)
            if block === nothing
                _checkblocksize(t.connectors, size(block_b), sectors)
                t.components[sectors] = f_unary(block_b)
            else
                broadcast!(f_binary, block, block, block_b)
            end
        end
    end
    return t
end

function Base.:+(a::Tensor{Ta}, b::Tb) where {Ta <: Number, Tb <: Number}
    t = Tensor{promote_type(Ta, Tb)}(a)
    for block in values(t.components)
        block .+= b
    end
    return t
end
function Base.:+(a::Ta, b::Tensor{Tb}) where {Ta <: Number, Tb <: Number}
    t = Tensor{promote_type(Ta, Tb)}(b)
    for block in values(t.components)
        broadcast!(+, block, a, block)
    end
    return t
end
Base.:+(a::Tensor, b::Number, bs::Number...) = a + (b + sum(bs))
Base.:+(a::Tensor, bs::Tensor...) = _map(+, x -> x, a, bs...)

function Base.:-(a::Tensor{Ta}, b::Tb) where {Ta <: Number, Tb <: Number}
    t = Tensor{promote_type(Ta, Tb)}(a)
    for block in values(t.components)
        block .-= b
    end
    return t
end
function Base.:-(a::Ta, b::Tensor{Tb}) where {Ta <: Number, Tb <: Number}
    t = Tensor{promote_type(Ta, Tb)}(b)
    for block in values(t.components)
        broadcast!(-, block, a, block)
    end
    return t
end
Base.:-(a::Tensor, b::Tensor) = _map(-, x -> -x, a, b)

function _arrangecomps(
    comps::Dict{NTuple{N, S}, Array{T, N}}, 
    inner::NTuple{M, Int}, 
    outer::NTuple{NminusM, Int}
) where {T <: Number, S <: SymmetrySector, N, M, NminusM}
    @assert N == M + NminusM
    inner = collect(inner)
    outer = collect(outer)
    perm = outer..., inner...
    CompVec = Vector{Pair{NTuple{NminusM, S}, Array{T, N}}}
    arcomps = DefaultDict{NTuple{M, S}, CompVec}(CompVec)
    for (sectors, compsblock) in pairs(comps)
        push!(arcomps[sectors[inner]], sectors[outer] => permutedims(compsblock, perm))
    end
    return arcomps
end

function _mulblocks(
    arcomps::DefaultDict{NTuple{M, S}, Vector{Pair{NTuple{Nleft, S}, Array{Ta, Na}}}},
    comps::Dict{NTuple{Nb, S}, Array{Tb, Nb}},
    inner::NTuple{M, Int},
    outer::NTuple{Nright, Int}, 
) where {Ta <: Number, Tb <: Number, S <: SymmetrySector, M, Na, Nb, Nleft, Nright}
    @assert Na == Nleft + M
    @assert Nb == Nright + M
    inner = collect(inner)
    outer = collect(outer)
    perm = inner..., outer...
    N = Nleft + Nright
    T = promote_type(Ta, Tb)
    mulcomps = Dict{NTuple{N, S}, Array{T, N}}()
    for (sectors, rightblock) in comps
        rightblock = permutedims(rightblock, perm)
        rightsize = size(rightblock)[begin + M : end]
        rightmatrix = reshape(rightblock, :, prod(rightsize))
        rightsecs = sectors[outer]
        for (leftsecs, leftblock) in arcomps[sectors[inner]]
            leftsize = size(leftblock)[begin : end - M]
            leftmatrix = reshape(leftblock, prod(leftsize), :)
            secs = tuple(leftsecs..., rightsecs...)
            block = reshape(leftmatrix * rightmatrix, leftsize..., rightsize...)
            resultblock = get!(mulcomps, secs, block)
            if resultblock ≢ block  # resultblock not set by previous line
                resultblock .+= block
            end
        end
    end
    return T, N, mulcomps
end

function Base.:*(
    a::Tensor{Ta, S}, b::Tensor{Tb, S}
) where {Ta <: Number, Tb <: Number, S <: SymmetrySector}
    inds_a, inds_b, m = matchconnectors(a.connectors, b.connectors)
    outer_a = @view inds_a[begin + m : end]
    inner_a = @view inds_a[begin : begin + m - 1]
    outer_b = @view inds_b[begin + m : end]
    inner_b = @view inds_b[begin : begin + m - 1]
    arcomps = _arrangecomps(a.components, Tuple(inner_a), Tuple(outer_a))
    T, N, comps = _mulblocks(arcomps, b.components, Tuple(inner_b), Tuple(outer_b))
    connects = tuple(a.connectors[outer_a]..., b.connectors[outer_b]...)
    return Tensor{T, S, N}(comps, connects, check = false)
end


DimsVector{S, N} = Vector{Pair{NTuple{N, S}, Int}}
function combinesectors_dims(
    connectors::NTuple{N, Connector{S}}
) where {S <: SymmetrySector, N}
    combsectordims = DefaultDict{S, DimsVector{S, N}}(DimsVector{S, N})
    sectors = Vector{S}(undef, N)
    combsectors = Vector{S}(undef, N)
    combdims = Vector{Int}(undef, N)
    iterstates = Vector{Union{Nothing, Int}}(nothing, N)
    k = 1
    while k > 0
        if k > N
            push!(combsectordims[combsectors[N]], Tuple(sectors) => combdims[N])
            k -= 1
            continue
        end
        con = connectors[k]
        dims = con.connection.dims
        state = iterstates[k]
        iter = state ≡ nothing ? iterate(dims) : iterate(dims, state)
        if iter ≡ nothing
            iterstates[k] = nothing
            k -= 1
            continue
        end
        (sector, dim), state = iter
        iterstates[k] = state
        sectors[k] = sector
        if !con.out
            sector = -sector
        end
        combsectors[k] = k == 1 ? sector : combsectors[k-1] + sector
        combdims[k] = k == 1 ? dim : combdims[k-1] * dim
        k += 1
    end
    return combsectordims
end

Arrangement{S, N} = Dict{NTuple{N, S}, Tuple{S, UnitRange{Int}}}
struct ConnectorCombinantion{S <: SymmetrySector, N}
    connector::Connector{S}
    arrangement::Arrangement{S, N}
    connectors::NTuple{N, Connector{S}}
end

function buildarrangement(
    combsectordims::DefaultDict{S, DimsVector{S, N}}, out::Bool
) where {S <: SymmetrySector, N}
    arrangement = Arrangement{S, N}()
    combdims = Dict{S, Int}()
    for (combsect, sectordims) in combsectordims
        if !out
            combsect = -combsect
        end
        sort!(sectordims, by=first)
        stop = 0
        for (sectors, dim) in sectordims
            start = stop + 1
            stop += dim
            arrangement[sectors] = combsect, start : stop
        end
        combdims[combsect] = stop
    end
    return arrangement, combdims
end

function mergeconnectors(
    connectors::Tuple{Connector{S}, Vararg{Connector{S}}}
) where S <: SymmetrySector
    name = Symbol((c.connection.name for c in connectors)...)
    tags = mergewith(Set ∘ union, (c.connection.tags for c in connectors)...)
    balance = sum(c.out ? +1 : -1 for c in connectors)
    out = balance == 0 ? first(connectors).out : balance > 0
    combsectordims = combinesectors_dims(connectors)
    arrangement, combdims = buildarrangement(combsectordims, out)
    connector = Connector(Connection{S}(combdims, name, tags); out)
    return ConnectorCombinantion(connector, arrangement, connectors)
end
mergeconnectors(connectors::Vararg{Connector}) = mergeconnectors(connectors)

function mergeconnectors(
    t::Tensor{T, S}, combines...
) where {T <: Number, S <: SymmetrySector}
    combranges = accumulate(combines, init=1:0) do r, cs
        s = r.stop
        s + 1 : s + length(cs)
    end
    connects = Tuple(flatten(combines))
    conrange, perm, m = matchconnectors(connects, t.connectors, direct = true)
    (m == length(connects) && conrange == 1:m) || 
        throw(ArgumentError("Connector to combine is not part of the Tensor"))
    remrange_old = length(connects) + 1 : rank(t)
    Ncomb = length(combines)
    N = Ncomb + length(remrange_old)
    remrange_new = Ncomb + 1 : N
    concombs = map(comb -> mergeconnectors(comb...), combines)
    comps = Dict{NTuple{N, S}, Array{T, N}}()
    combsects = Vector{S}(undef, N)
    ranges = Vector{UnitRange{Int}}(undef, N)
    fulldims = Vector{Int}(undef, N)
    for (sectors, block) in t.components
        permsects = sectors[perm]
        permblock = permutedims(block, perm)
        for (k, combrange) in pairs(combranges)
            combsects[k], ranges[k] = concombs[k].arrangement[permsects[combrange]]
        end
        combsects[remrange_new] .= permsects[remrange_old]
        ranges[remrange_new] .= axes.(Ref(permblock), remrange_old)
        combblock = reshape(permblock, length.(ranges)...)
        fullblock = get!(comps, Tuple(combsects)) do 
            for (k, sector) in pairs(@view combsects[1:Ncomb])
                fulldims[k] = concombs[k].connector.connection.dims[sector]
            end
            fulldims[remrange_new] .= length.(@view ranges[remrange_old])
            zeros(T, fulldims...)
        end
        fullblock[ranges...] = combblock
    end
    combconnects = tuple(
        getfield.(concombs, :connector)..., 
        t.connectors[perm[remrange_old]]...
    )
    return Tensor{T, S, N}(comps, combconnects, check = false), concombs
end


RangeSectors{S, N} = Vector{Pair{UnitRange{Int}, NTuple{N, S}}}
function invertarrangement(arrangement::Arrangement{S, N}) where {S <: SymmetrySector, N}
    rangesectors = DefaultDict{S, RangeSectors{S, N}}(RangeSectors{S, N})
    for (sectors, (sector, range)) in arrangement
        push!(rangesectors[sector], range => sectors)
    end
    return rangesectors
end

function separateconnectors(
    t::Tensor{T, S, N}, separations::ConnectorCombinantion{S}...
) where {T <: Number, S <: SymmetrySector, N}
    rangesectors = invertarrangement.(getfield.(separations, :arrangement))
    conrange, pos, m = matchconnectors(
        getfield.(separations, :connector), t.connectors, direct = true
    )
    m == length(conrange) || throw(
        ArgumentError("Connector to separate is not part of the Tensor")
    )
    @assert conrange == 1:m
    seppos = pos[begin : m]
    rempos = pos[m + 1 : end]
    seplengths = ones(Int, N)
    seplengths[seppos] .= length.(getfield.(separations, :connectors))
    sectranges = accumulate(seplengths, init=1:0) do r, l
        s = r.stop
        s + 1 : s + l
    end
    @assert all(r -> r.start == r.stop, sectranges[rempos])
    rempos_new = getfield.(view(sectranges, rempos), :start)
    sepranges = sectranges[seppos]
    Nsep = sectranges[end].stop
    connectors = Vector{Connector{S}}(undef, Nsep)
    connectors[rempos_new] .= t.connectors[rempos]
    for (k, seprange) in pairs(sepranges)
        connectors[seprange] .= separations[k].connectors
    end
    comps = Dict{NTuple{Nsep, S}, Array{T, Nsep}}()
    ranges = Vector{Union{Colon, UnitRange{Int}}}(undef, N)
    ranges[rempos] .= Colon()
    sepsectors = Vector{S}(undef, Nsep)
    sepdims = Vector{Int}(undef, Nsep)
    for (sectors, block) in t.components
        sepsectors[rempos_new] .= sectors[rempos]
        sepdims[rempos_new] .= size(block)[rempos]
        rangesectvects = (rangesectors[k][sectors[seppos[k]]] for k = 1:m)
        for rangesects in product(rangesectvects...)
            for (k, (range, sects)) in pairs(rangesects)
                ranges[seppos[k]] = range
                sepsectors[sepranges[k]] .= sects
                connects = separations[k].connectors
                sepdims[sepranges[k]] .= getindex.(
                    getfield.(getfield.(connects, :connection), :dims), sects
                )
            end
            subblock = block[ranges...]
            if !iszero(subblock)
                comps[Tuple(sepsectors)] = reshape(subblock, sepdims...)
            end
        end
    end
    Tensor{T, S, Nsep}(comps, Tuple(connectors), check = false)
end


function svd(
    t::Tensor{T, S}, splitoff::Connector{S}...; maxblockdim::Int = typemax(Int)
) where {T <: Number, S <: SymmetrySector}
    remaining = setdiff(t.connectors, splitoff)
    tmatrix, (combo_split, combo_remain) = mergeconnectors(t, splitoff, remaining)
    Ucomps = Dict{NTuple{2, S}, Array{T, 2}}()
    Scomps = Dict{NTuple{2, S}, Array{T, 2}}()
    Vcomps = Dict{NTuple{2, S}, Array{T, 2}}()
    total = mergesectors(first(keys(tmatrix.components)), tmatrix.connectors)
    for ((sector_split, sector_remain), block) in tmatrix.components
        @assert total == mergesectors((sector_split, sector_remain), tmatrix.connectors)
        F = svd!(block)
        cutoff = min(length(F.S), maxblockdim)
        Ucomps[(sector_split, sector_split)] = F.U[:, begin:cutoff]
        Scomps[(sector_split, sector_split)] = Diagonal(F.S[begin:cutoff])
        Vcomps[(sector_split, sector_remain)] = F.Vt[begin:cutoff, :]
    end
    Uconnector = replicate(combo_split.connector)
    Vconnector = replicate(Uconnector)
    Umatrix = Tensor(Ucomps, combo_split.connector, dual(Uconnector))
    Stensor = Tensor(Scomps, Uconnector, dual(Vconnector))
    Vmatrix = Tensor(Vcomps, Vconnector, combo_remain.connector)
    Utensor = separateconnectors(Umatrix, combo_split)
    Vtensor = separateconnectors(Vmatrix, combo_remain)
    return Utensor, Stensor, Vtensor
end

function qr(
    t::Tensor{T, S}, splitoff::Connector{S}...
) where {T <: Number, S <: SymmetrySector}
    remaining = setdiff(t.connectors, splitoff)
    tmatrix, (combo_split, combo_remain) = mergeconnectors(t, splitoff, remaining)
    Qcomps = Dict{NTuple{2, S}, Array{T, 2}}()
    Rcomps = Dict{NTuple{2, S}, Array{T, 2}}()
    total = mergesectors(first(keys(tmatrix.components)), tmatrix.connectors)
    for ((sector_split, sector_remain), block) in tmatrix.components
        @assert total == mergesectors((sector_split, sector_remain), tmatrix.connectors)
        F = qr!(block)
        Qcomps[(sector_split, sector_remain)] = F.Q
        Rcomps[(sector_remain, sector_remain)] = F.R
    end
    connector = replicate(combo_split.connector)
    Qmatrix = Tensor(Qcomps, combo_split.connector, dual(connector))
    Rmatrix = Tensor(Rcomps, connector, combo_remain.connector)
    Qtensor = separateconnectors(Qmatrix, combo_split)
    Rtensor = separateconnectors(Rmatrix, combo_remain)
    return Qtensor, Rtensor
end

function lq(
    t::Tensor{T, S}, splitoff::Connector{S}...
) where {T <: Number, S <: SymmetrySector}
    remaining = setdiff(t.connectors, splitoff)
    tmatrix, (combo_remain, combo_split) = mergeconnectors(t, remaining, splitoff)
    Lcomps = Dict{NTuple{2, S}, Array{T, 2}}()
    Qcomps = Dict{NTuple{2, S}, Array{T, 2}}()
    total = mergesectors(first(keys(tmatrix.components)), tmatrix.connectors)
    for ((sector_remain, sector_split), block) in tmatrix.components
        @assert total == mergesectors((sector_split, sector_remain), tmatrix.connectors)
        F = lq!(block)
        Lcomps[(sector_remain, sector_remain)] = F.L
        Qcomps[(sector_remain, sector_split)] = F.Q
    end
    connector = replicate(combo_split.connector)
    Lmatrix = Tensor(Lcomps, combo_remain.connector, connector)
    Qmatrix = Tensor(Qcomps, dual(connector), combo_split.connector)
    Ltensor = separateconnectors(Lmatrix, combo_remain)
    Qtensor = separateconnectors(Qmatrix, combo_split)
    return Ltensor, Qtensor
end

end