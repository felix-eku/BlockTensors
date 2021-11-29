function buildsectordims(
    components::AbstractDict{NTuple{N, S}, <:AbstractArray{T, N}}
) where {T, S <: SymmetrySector, N}
    dims = [Dict{S, Int}() for k = 1:N]
    for dict in dims
        sizehint!(dict, length(components))
    end
    for (sectors, block) in components
        blocksize = size(block)
        for k = 1:N
            blocksize[k] == get!(dims[k], sectors[k], blocksize[k]) || throw(
                DimensionMismatch("multiple dimensions for sector $(sectors[k]) of leg $k")
            )
        end
    end
    return SectorDims.(dims)
end

function checkblockdims(
    legs::NTuple{N, Leg{S}}, sectors::NTuple{N, S}, blocksize::NTuple{N, Int}
) where {S <: SymmetrySector, N}
    for k = 1:N
        blocksize[k] == get(legs[k].dimensions, sectors[k], -1) || throw(
            DimensionMismatch(
                "wrong block dimension for sector $(sectors[k]) of leg $(legs[k])"
            )
        )
    end
end

struct Tensor{T <: Number, S <: SymmetrySector, N}
    components::Dict{NTuple{N, S}, Array{T, N}}
    legs::NTuple{N, Leg{S}}
    function Tensor{T}(
        components::Dict{NTuple{N, S}, Array{T, N}}, legs::NTuple{N, Leg{S}};
        check::Bool = true
    ) where {T <: Number, S <: SymmetrySector, N}
        if check
            allunique(leg.connector for leg in legs) || throw(
                ArgumentError("multiple legs for the same connector")
            )
            for (sectors, block) in components
                checkblockdims(legs, sectors, size(block))
            end
        end
        new{T, S, N}(components, legs)
    end
end
function Tensor{T}(
    components::Dict{Tuple{}, Array{T, 0}}, legs::Tuple{}; check = true
) where T <: Number
    components[()][]
end

function Tensor{T}(
    t::Tensor{Tprime, S, N}
) where {T <: Number, Tprime <: Number, S <: SymmetrySector, N}
    Tensor{T}(
        Dict(sector => Array{T, N}(block) for (sector, block) in t.components),
        t.legs, check = false
    )
end
Tensor(t::Tensor{T}) where T <: Number = Tensor{T}(t)

LegLike{S <: SymmetrySector} = Union{Leg{S}, <:Connector}
function Tensor{T}(
    components::Dict{NTuple{N, S}, Array{T, N}}, legs::NTuple{N, LegLike{S}}, 
    dims = buildsectordims(components)
) where {T <: Number, S <: SymmetrySector, N}
    legs = Tuple(map(Leg, legs, dims))
    Tensor{T}(components, legs, check = false)
end
Components{T <: Number, S <: SymmetrySector, N} = Union{
    AbstractDict{NTuple{N, S}, <:AbstractArray{T, N}}, AbstractArray{T, N}
}
function Tensor{T}(
    components::Components{Tprime, S, N}, legs::Vararg{LegLike{S}, N}
) where {T <: Number, Tprime <: Number, S <: SymmetrySector, N}
    Tensor{T}(components, legs)
end
function Tensor(
    components::Components{T, S, N}, legs::NTuple{N, LegLike{S}}
) where {T <: Number, S <: SymmetrySector, N}
    Tensor{T}(components, legs)
end
function Tensor(
    components::Components{T, S, N}, legs::Vararg{LegLike{S}, N}
) where {T <: Number, S <: SymmetrySector, N}
    Tensor{T}(components, legs)
end

function Tensor{T}(
    components::AbstractDict{NTuple{N, S}, <:AbstractArray{Tprime, N}},
    legs::NTuple{N, LegLike{S}}
) where {T <: Number, Tprime <: Number, S <: SymmetrySector, N}
    Tensor{T}(convert(Dict{NTuple{N, S}, Array{T, N}}, components), legs)
end
function Tensor{T}(
    components::AbstractArray{Tprime, N},
    legs::NTuple{N, Leg{Trivial}}
) where {T <: Number, Tprime <: Number, N}
    sectors = ntuple(_ -> Trivial(), Val(N))
    Tensor{T}(Dict(sectors => convert(Array{T, N}, components)), legs)
end

function Base.copy(t::Tensor{T}) where T <: Number
    Tensor{T}(t.components, copy.(t.legs), check = false)
end

Base.eltype(::Type{<:Tensor{T}}) where T <: Number = T

function Base.adjoint(t::Tensor{T}) where T <: Number
    Tensor{T}(
        Dict(sectors => conj(block) for (sectors, block) in t.components),
        dual.(t.legs, connect = true),
        check = false
    )
end

function Base.show(io::IO, mime::MIME"text/plain", t::Tensor)
    show(io, typeof(t)); println(io, " with legs:")
    for leg in t.legs
        print(io, "  "); show(io, mime, leg); println(io)
    end
    print(io, "blocks:")
    for (sectors, block) in t.components
        println(io); print(io, "  "); show(IOContext(io, :compact => true), sectors)
        print(io, " => "); join(io, size(block), "×"); print(io, " block")
    end
end

function Base.show(io::IO, t::Tensor)
    TensorT = typeof(t)
    show(io, (:typeinfo => TensorT) in io ? Tensor : TensorT)
    print(io, "(")
    join(io, getfield.(t.legs, :connector), ", ")
    print(io, ")")
end


rank(::Tensor{T, S, N}) where {T <: Number, S <: SymmetrySector, N} = N

function legspermutation(
    t::Tensor{T, S, N}, legs::NTuple{N, LegLike{S}}
) where {T <: Number, S <: SymmetrySector, N}
    identperm, perm, n = matchingpermutations(t.legs, legs)
    n == N || throw(
        ArgumentError("$legs does not represent a permutation of the legs of the tensor")
    )
    @assert identperm == 1:N
    return perm
end

function _get_sectors_indices(::Type{Trivial}, blockinds::NTuple{N}) where N
    sectors = ntuple(_ -> Trivial(), Val(N))
    return sectors, blockinds
end
function _get_sectors_indices(
    ::Type{S}, blockinds::Tuple{Vararg{Pair{S, <:Any}}}
) where S <: SymmetrySector
    sectors = getproperty.(blockinds, :first)
    inds = getproperty.(blockinds, :second)
    return sectors, inds
end
function _get_sectors_indices(
    ::Type{S}, sectors::Tuple{Vararg{S}}
) where S <: SymmetrySector
    inds = fill(:, length(sectors))
    return sectors, inds
end

function Base.getindex(
    t::Tensor{T, S, N},
    indices::Vararg{Pair{<: LegLike{S}, <: Any}, N}
) where {T <: Number, S <: SymmetrySector, N}
    legs = getproperty.(indices, :first)
    perm = legspermutation(t, legs)
    blockinds = getproperty.(indices, :second)
    sectors, inds = _get_sectors_indices(S, blockinds)
    return PermutedDimsArray(t.components[sectors[perm]], invperm(perm))[inds...]
end

matching(x, t::Tensor) = matching(x, t.legs)
matchingpositions(x, t::Tensor) = findall(matching(x), t.legs)

matchlegs(t::Tensor, leglikes...) = only.(matching.(leglikes, Ref(t)))


function connect!(a::Tensor, b::Tensor, connecting)
    connect!(a, b, Outgoing(connecting), Incoming(connecting))
end
function connect!(a::Tensor, b::Tensor, outgoing::Outgoing, incoming::Incoming)
    lout = only(matchingpositions(outgoing, a))
    lin = only(matchingpositions(incoming, b))
    id = connect!(a.legs[lout], b.legs[lin])
    return id, lout, lin
end
function connect!(a::Tensor, b::Tensor, connecting, lout::Integer, lin::Integer)
    connect!(a, b, Outgoing(connecting), Incoming(connecting), lout, lin)
end
function connect!(
    a::Tensor, b::Tensor, outgoing::Outgoing, incoming::Incoming, lout::Integer, lin::Integer
)
    legout = get(a.legs, lout, nothing)
    legin = get(b.legs, lin, nothing)
    if matching(outgoing, legout) && matching(incoming, legin)
        id = connect!(legout, legin)
        return id, lout, lin
    else
        connect!(a, b, outgoing, incoming)
    end
end


function prune!(t::Tensor)
    filter!(sectors_block -> !iszero(sectors_block.second), t.components)
    return nothing
end

function _map(
    f_binary, f_unary, a::Tensor{Ta, S, N}, bs::Vararg{Tensor{Tb, S, N} where Tb <: Number}
) where {Ta <: Number, S <: SymmetrySector, N}
    T = promote_type(Ta, map(TensorT -> TensorT.parameters[1], typeof(bs).parameters)...)
    t = Tensor{T}(a)
    for b in bs
        perm = legspermutation(t, b.legs)
        for (sectors_perm, block_b) in b.components
            sectors = sectors_perm[perm]
            block_b = PermutedDimsArray(block_b, perm)
            block = get(t.components, sectors, nothing)
            if block ≡ nothing
                checkblockdims(t.legs, sectors, size(block_b))
                t.components[sectors] = f_unary(block_b)
            else
                broadcast!(f_binary, block, block, block_b)
            end
        end
    end
    return t
end

for op = (:+, :-, :*, :/)
    eval(quote
        function Base.$op(a::Tensor{Ta}, b::Tb) where {Ta <: Number, Tb <: Number}
            t = Tensor{promote_type(Ta, Tb)}(a)
            for block in values(t.components)
                broadcast!($op, block, block, b)
            end
            return t
        end
    end)
end
for op = (:+, :-, :*, :\)
    eval(quote
        function Base.$op(a::Ta, b::Tensor{Tb}) where {Ta <: Number, Tb <: Number}
            t = Tensor{promote_type(Ta, Tb)}(b)
            for block in values(t.components)
                broadcast!($op, block, a, block)
            end
            return t
        end
    end)
end
for op = (:+, :*)
    eval(quote
        Base.$op(a::Tensor, b::Number, bs::Number...) = $op(a, $op(b, bs...))
    end)
end

Base.:+(a::Tensor, bs::Tensor...) = _map(+, x -> x, a, bs...)
Base.:-(a::Tensor, b::Tensor) = _map(-, x -> -x, a, b)

function _arrangecomponents(
    comps::Dict{NTuple{N, S}, Array{T, N}},
    inner::NTuple{M, Int},
    outer::NTuple{NminusM, Int}
) where {T <: Number, S <: SymmetrySector, N, M, NminusM}
    @assert N == M + NminusM
    inner = collect(inner)
    outer = collect(outer)
    perm = outer..., inner...
    ArrangementVector = Vector{Pair{NTuple{NminusM, S}, Array{T, N}}}
    arrangement = DefaultDict{NTuple{M, S}, ArrangementVector}(ArrangementVector)
    for (sectors, block) in pairs(comps)
        push!(arrangement[sectors[inner]], sectors[outer] => permutedims(block, perm))
    end
    return arrangement
end

function _multiplyblocks(
    arrangement::DefaultDict{NTuple{M, S}, Vector{Pair{NTuple{Nleft, S}, Array{Ta, Na}}}},
    components::Dict{NTuple{Nb, S}, Array{Tb, Nb}},
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
    comps = Dict{NTuple{N, S}, Array{T, N}}()
    for (sectors, rightblock) in components
        rightblock = permutedims(rightblock, perm)
        rightsize = size(rightblock)[begin + M : end]
        rightmatrix = reshape(rightblock, :, prod(rightsize))
        rightsectors = sectors[outer]
        for (leftsectors, leftblock) in arrangement[sectors[inner]]
            leftsize = size(leftblock)[begin : end - M]
            leftmatrix = reshape(leftblock, prod(leftsize), :)
            secs = tuple(leftsectors..., rightsectors...)
            block = reshape(leftmatrix * rightmatrix, leftsize..., rightsize...)
            resultblock = get!(comps, secs, block)
            if resultblock ≢ block  # resultblock not set by previous line
                resultblock .+= block
            end
        end
    end
    return T, comps
end

function Base.:*(
    a::Tensor{Ta, S}, b::Tensor{Tb, S}
) where {Ta <: Number, Tb <: Number, S <: SymmetrySector}
    inds_a, inds_b, m = matchingpermutations(a.legs, b.legs, matchs = connected)
    outer_a = @view inds_a[begin + m : end]
    inner_a = @view inds_a[begin : begin + m - 1]
    outer_b = @view inds_b[begin + m : end]
    inner_b = @view inds_b[begin : begin + m - 1]
    arrangement = _arrangecomponents(a.components, Tuple(inner_a), Tuple(outer_a))
    T, comps = _multiplyblocks(arrangement, b.components, Tuple(inner_b), Tuple(outer_b))
    legs = tuple(a.legs[outer_a]..., b.legs[outer_b]...)
    return Tensor{T}(comps, legs, check = false)
end


@inline flip(@nospecialize f) = (a, b) -> f(b, a)

function mergelegs(
    t::Tensor{T, S}, legiterators...
) where {T <: Number, S <: SymmetrySector}
    Nmerge = 0
    mergers = [
        (Nmerge + 1) : (Nmerge += length(merger)) 
        for merger in legiterators if length(merger) > 0
    ]
    leglikes = Tuple(Iterators.flatten(legiterators))
    identperm, perm, m = matchingpermutations(leglikes, t.legs, matchs = flip(matching))
    (m == Nmerge && identperm == 1:m) ||
        throw(ArgumentError("tensor has not all of the legs to merge"))
    keep = Nmerge + 1 : rank(t)
    Nmerged = length(mergers)
    N = Nmerged + length(keep)
    kept = Nmerged + 1 : N
    mergedlegs = map(merger -> Leg(t.legs[perm[merger]]), mergers)
    comps = Dict{NTuple{N, S}, Array{T, N}}()
    mergedsectors = Vector{S}(undef, N)
    ranges = Vector{UnitRange{Int}}(undef, N)
    fulldims = Vector{Int}(undef, N)
    for (sectors, block) in t.components
        permsectors = sectors[perm]
        permblock = permutedims(block, perm)
        for (k, merger) in pairs(mergers)
            mergedsectors[k], ranges[k] = mergedlegs[k].arrangement[permsectors[merger]]
        end
        mergedsectors[kept] .= permsectors[keep]
        ranges[kept] .= axes.(Ref(permblock), keep)
        mergedblock = reshape(permblock, length.(ranges)...)
        fullblock = get!(comps, Tuple(mergedsectors)) do
            for (k, sector) in pairs(@view mergedsectors[1:Nmerged])
                fulldims[k] = mergedlegs[k].dimensions[sector]
            end
            fulldims[kept] .= length.(@view ranges[kept])
            zeros(T, fulldims...)
        end
        fullblock[ranges...] = mergedblock
    end
    legs = tuple(mergedlegs..., t.legs[perm[keep]]...)
    return Tensor{T}(comps, legs, check = false)
end


function separatelegs(
    t::Tensor{T, S, N}, leglikes::LegLike{S}...
) where {T <: Number, S <: SymmetrySector, N}
    identperm, perm, m = matchingpermutations(leglikes, t.legs, matchs = flip(matching))
    m == length(identperm) || throw(
        ArgumentError("tensor has not all of the legs to separate")
    )
    @assert identperm == 1:m
    separate = perm[begin : m]
    keep = perm[m + 1 : end]
    legs = t.legs[separate]
    componentleg_counts = ones(Int, N)
    componentleg_counts[separate] .= length.(getfield.(legs, :components))
    Nsep = 0
    separated = [(Nsep + 1) : (Nsep += count) for count in componentleg_counts]
    @assert all(r -> r.start == r.stop, separated[keep])
    kept = getfield.(view(separated, keep), :start)
    separated = separated[separate]
    rangesectors = invertarrangement.(getfield.(legs, :arrangement))
    comps = Dict{NTuple{Nsep, S}, Array{T, Nsep}}()
    ranges = Vector{Union{Colon, UnitRange{Int}}}(undef, N)
    ranges[keep] .= Colon()
    sepsectors = Vector{S}(undef, Nsep)
    sepdims = Vector{Int}(undef, Nsep)
    for (sectors, block) in t.components
        sepsectors[kept] .= sectors[keep]
        sepdims[kept] .= size.(Ref(block), keep)
        range_sector_iterators = (rangesectors[k][sectors[separate[k]]] for k = 1:m)
        for range_sector_product in product(range_sector_iterators...)
            for (k, (range, subsectors)) in pairs(range_sector_product)
                ranges[separate[k]] = range
                sepsectors[separated[k]] .= subsectors
                sepdims[separated[k]] .= getindex.(
                    getfield.(legs[k].components, :dimensions), subsectors
                )
            end
            subblock = block[ranges...]
            if !iszero(subblock)
                comps[Tuple(sepsectors)] = reshape(subblock, sepdims...)
            end
        end
    end
    seplegs = Vector{Leg{S}}(undef, Nsep)
    seplegs[kept] .= t.legs[keep]
    for (k, range) in pairs(separated)
        seplegs[range] .= legs[k].components
    end
    Tensor{T}(comps, Tuple(seplegs), check = false)
end


function unmatched(
    legs::NTuple{N, Leg{S}}, leglikes::Tuple{Vararg{LegLike{S}}}
) where {S <: SymmetrySector, N}
    unmatched = fill(true, N)
    for leglike in leglikes
        for (k, leg) in pairs(legs)
            unmatched[k] || continue
            if matching(leglike, leg)
                unmatched[k] = false
                break
            end
        end
    end
    return legs[unmatched]
end

function svd(
    t::Tensor{T, S}, splitoff::LegLike{S}...; 
    maxrelerror = zero(real(T)),
    maxabserror = zero(real(T)),
    of = x -> x^2,
    maxdim = typemax(Int),
) where {T <: Number, S <: SymmetrySector}
    remaining = unmatched(t.legs, splitoff)
    tmatrix = mergelegs(t, remaining, splitoff)
    prune!(tmatrix)
    singulars = Vector{real(T)}()
    sizehint!(singulars, minimum(leg -> leg.dimensions.totaldim, tmatrix.legs))
    Ucomps = Dict{NTuple{2, S}, Array{T, 2}}()
    Scomps = Dict{NTuple{2, S}, Diagonal{real(T)}}()
    Vcomps = Dict{NTuple{2, S}, Array{T, 2}}()
    total = combine(getfield.(tmatrix.legs, :connector), first(keys(tmatrix.components)))
    for (sectors, block) in tmatrix.components
        @assert total == combine(getfield.(tmatrix.legs, :connector), sectors)
        F = svd!(block)
        append!(singulars, F.S)
        sector_remain, sector_split = sectors
        Ucomps[sector_remain, sector_split] = F.U
        Scomps[sector_split, sector_split] = Diagonal(F.S)
        Vcomps[sector_split, sector_split] = F.Vt
    end
    sort!(singulars, rev = true)
    λs = of.(singulars)
    λtotal = sum(λs)
    errorcutoff = max(maxrelerror * λtotal, maxabserror)
    error = zero(eltype(λs))
    local λ
    for outer λ in Iterators.reverse(λs)
        error += λ
        error < errorcutoff || break
    end
    error -= λ
    maxdim > lastindex(λs) || (λ = max(λ, λs[maxdim]))
    for ((sector_remain, sector_split), Ublock) in Ucomps 
        Vblock = Vcomps[sector_split, sector_split]
        singulars_sector = diag(Scomps[sector_split, sector_split])
        cutoff = findlast(s -> of(s) ≥ λ, singulars_sector)
        if cutoff ≢ nothing
            Ucomps[sector_remain, sector_split] = Ublock[:, begin:cutoff]
            Scomps[sector_split, sector_split] = Diagonal(singulars_sector[begin:cutoff])
            Vcomps[sector_split, sector_split] = Vblock[begin:cutoff, :]
        else
            delete!(Ucomps, (sector_remain, sector_split))
            delete!(Scomps, (sector_split, sector_split))
            delete!(Vcomps, (sector_split, sector_split))
        end
    end
    leg_remain, leg_split = tmatrix.legs
    Umatrix = Tensor(Ucomps, leg_remain, leg_split.connector)
    Uinner = Umatrix.legs[2]
    Vinner = copy(Uinner)
    Uinner.connection.id = nextid()
    Vinner.connection.id = nextid()
    Stensor = Tensor(Scomps, dual(Uinner, connect = true), Vinner)
    Vmatrix = Tensor(Vcomps, dual(Vinner, connect = true), leg_split)
    Utensor = separatelegs(Umatrix, leg_remain)
    Vtensor = separatelegs(Vmatrix, leg_split)
    return Utensor, Stensor, Vtensor, singulars, error / λtotal, λ, λtotal
end

function qr(
    t::Tensor{T, S}, splitoff::LegLike{S}...
) where {T <: Number, S <: SymmetrySector}
    remaining = unmatched(t.legs, splitoff)
    tmatrix = mergelegs(t, remaining, splitoff)
    prune!(tmatrix)
    Qcomps = Dict{NTuple{2, S}, Array{T, 2}}()
    Rcomps = Dict{NTuple{2, S}, Array{T, 2}}()
    total = combine(getfield.(tmatrix.legs, :connector), first(keys(tmatrix.components)))
    for (sectors, block) in tmatrix.components
        @assert total == combine(getfield.(tmatrix.legs, :connector), sectors)
        F = qr!(block)
        (sector_remain, sector_split) = sectors
        Qcomps[sector_remain, sector_split] = F.Q
        Rcomps[sector_split, sector_split] = F.R
    end
    leg_remain, leg_split = tmatrix.legs
    Qmatrix = Tensor(Qcomps, leg_remain, leg_split.connector)
    inner = Qmatrix.legs[2]
    inner.connection.id = nextid()
    Rmatrix = Tensor(Rcomps, dual(inner, connect = true), leg_split)
    Qtensor = separatelegs(Qmatrix, leg_remain)
    Rtensor = separatelegs(Rmatrix, leg_split)
    return Qtensor, Rtensor
end

function lq(
    t::Tensor{T, S}, splitoff::LegLike{S}...
) where {T <: Number, S <: SymmetrySector}
    remaining = unmatched(t.legs, splitoff)
    tmatrix = mergelegs(t, remaining, splitoff)
    prune!(tmatrix)
    Lcomps = Dict{NTuple{2, S}, Array{T, 2}}()
    Qcomps = Dict{NTuple{2, S}, Array{T, 2}}()
    total = combine(getfield.(tmatrix.legs, :connector), first(keys(tmatrix.components)))
    for (sectors, block) in tmatrix.components
        @assert total == combine(getfield.(tmatrix.legs, :connector), sectors)
        F = lq!(block)
        (sector_remain, sector_split) = sectors
        Lcomps[sector_remain, sector_split] = F.L
        Qcomps[sector_split, sector_split] = F.Q
    end
    leg_remain, leg_split = tmatrix.legs
    Lmatrix = Tensor(Lcomps, leg_remain, leg_split.connector)
    inner = Lmatrix.legs[2]
    inner.connection.id = nextid()
    Qmatrix = Tensor(Qcomps, dual(inner, connect = true), leg_split)
    Ltensor = separatelegs(Lmatrix, leg_remain)
    Qtensor = separatelegs(Qmatrix, leg_split)
    return Ltensor, Qtensor
end
