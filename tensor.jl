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


SymbolOrString = Union{Symbol, AbstractString}

struct Connection{S <: SymmetrySector}
    dims::Dict{S, Int}
    name::Symbol
    tags::Dict{Symbol, Any}
end
function Connection(
    dims::AbstractDict{S, <: Integer}, 
    name::SymbolOrString, 
    tags::Pair{<: SymbolOrString}...
) where S <: SymmetrySector
    Connection{S}(
        convert(Dict{S, Int}, dims), Symbol(name), 
        Dict(Symbol(label) => value for (label, value) in tags)
    )
end
function Connection{S}(name::SymbolOrString, tags...) where S <: SymmetrySector
    Connection(Dict{S, Int}(), name, tags...)
end
Connection(name::SymbolOrString, tags...) = Connection{Trivial}(name, tags...)
Connection{Trivial}(dim::Integer, name, tags...) = Connection(Dict(Trivial() => dim), name, tags...)
Connection(dim::Integer, name, tags...) = Connection{Trivial}(dim, name, tags...) 

struct Connector{S <: SymmetrySector}
    connection::Connection{S}
    out::Bool
end
Connector(connection::Connection; out::Bool) = Connector(connection, out)

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
    components::AbstractDict{NTuple{N, S}, AbstractArray{Tprime, N}},
    connectors::Vararg{Connector{S}, N}
) where {T <: Number, Tprime <: Number, S <: SymmetrySector, N}
    Tensor{T, S, N}(convert(Dict{NTuple{N, S}, Array{T, N}}, components), connectors)
end
function Tensor{T}(connectors::Vararg{Connector{S}, N}) where {T <: Number, S <: SymmetrySector, N}
    Tensor{T, S, N}(Dict{NTuple{N, S}, Array{T, N}}(), connectors, check = false)
end
function Tensor(
    components::AbstractDict{NTuple{N, S}, AbstractArray{T, N}}, 
    connectors::Vararg{Connector{S}, N}
) where {T <: Number, S <: SymmetrySector, N}
    Tensor{T}(components, connectors...)
end
function Tensor(components::Array{T, N}, connectors::Vararg{Connector{Trivial}, N}) where {T <: Number, N}
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

end