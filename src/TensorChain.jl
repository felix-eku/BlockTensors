module TensorChain

export chaintensors

using Base: setindex
using BlockTensors

struct Connection{N}
    positions::Pair{CartesianIndex{N}, CartesianIndex{N}}
    legs::Pair{Int}
end

@enum ConnnectionStatus::Int8 uncon oldcon newcon

function chaintensors(
    extratags, tensors::AbstractArray{<:Tensor, N}, connecting, 
    repeats::Integer, dim::Integer = N + 1
) where N
    0 ≤ dim ≤ N + 1 || throw(DomainError(dim))
    connections, connection_stati = findconnections(tensors, connecting, dim)
    Base.require_one_based_indexing(tensors)
    if dim == 0
        chained = similar(tensors, repeats, size(tensors)...)
        axis = axes(chained, 1)
        combine_indices = (k, I) -> CartesianIndex(k, I)
    elseif dim == N + 1
        chained = similar(tensors, size(tensors)..., repeats)
        axis = axes(chained, N + 1)
        combine_indices = (k, I) -> CartesianIndex(I, k)
    else
        chained = similar(tensors, setindex(size(tensors), size(tensors, dim) * repeats, dim))
        axis = (axes(chained, dim) .- 1)[begin : size(tensors, dim) : end]
        combine_indices = (k, I) -> setindex(I, I[dim] + k, dim)
    end
    for k in axis
        tags = extratags(k)
        for I in CartesianIndices(tensors)
            t = chained[combine_indices(k, I)] = copy(tensors[I])
            for leg in t.legs[connection_stati[I] .≢ newcon]
                addtags!(leg; tags...)
            end
        end
    end
    for k in axis
        for connection in connections
            I, J = connection.positions
            i, j = connection.legs
            if I == J
                k ≠ last(axis) || continue
                k_J = combine_indices(k + step(axis), I)
            else
                k_J = combine_indices(k, J)
                if 1 ≤ dim ≤ N
                    k_J[dim] ∈ axes(tensors, dim) || continue
                end
            end
            connect!(chained[combine_indices(k, I)].legs[i], chained[k_J].legs[j])
        end
    end
    return chained
end
function chaintensors(extratags, tensor::Tensor, connecting, repeats::Integer)
    chaintensors(extratags, fill(tensor, ()), connecting, repeats)
end

function findconnections(
    tensors::AbstractArray{<:Tensor, N}, connecting, dim::Integer = N + 1
) where N
    connection_stati = similar(tensors, Vector{ConnnectionStatus})
    for I in eachindex(tensors)
        connection_stati[I] = fill(uncon, length(tensors[I].legs))
    end
    connections = Vector{Connection{N}}()

    add_existingconnections!(tensors, connections, connection_stati)
    if !isempty(connecting)
        add_newconnections!(tensors, connecting, connections, connection_stati, dim)
    end
    add_existingconnections!(tensors, connections, connection_stati)
    return connections, connection_stati
end

function add_existingconnections!(
    tensors::AbstractArray{<:Tensor, N}, 
    connections::Vector{Connection{N}}, 
    connection_stati::Array{Vector{ConnnectionStatus}, N}
) where N
    R = CartesianIndices(tensors)
    Ilast = last(R)
    Izero = zero(Ilast)
    offsets = [Base.setindex(Izero, 1, k) for k = 1:N]
    for I in R, (i, leg_i) in pairs(tensors[I].legs)
        connection_stati[I][i] ≡ uncon || continue
        for (k, offset) in pairs(offsets)
            I[k] < Ilast[k] || continue
            J = I + offset
            for (j, leg_j) in pairs(tensors[J].legs)
                connection_stati[J][j] ≡ uncon || continue
                if connected(leg_i, leg_j)
                    push!(connections, Connection(I => J, i => j))
                    connection_stati[I][i] = connection_stati[J][j] = oldcon
                end
            end
        end
    end
    return nothing
end

function add_newconnections!(
    tensors::AbstractArray{<:Tensor, N}, connecting,
    connections::Vector{Connection{N}}, 
    connection_stati::Array{Vector{ConnnectionStatus}, N},
    dim::Integer = N + 1
) where N
    if dim < 1 || dim > N
        for I in CartesianIndices(tensors)
            add_newconnections!(tensors, connecting, connections, connection_stati, I)
        end
    else
        Rpre = axes(tensors)[begin : dim - 1]
        Rpost = axes(tensors)[dim + 1 : end]
        axis = axes(tensors, dim)
        Iin, Iout = first(axis), last(axis)
        Ioffset = Iout + oneunit(Iin)
        for Ipost in Rpost, Ipre in Rpre
            I = CartesianIndex(Ipre, Iout, Ipost)
            J = CartesianIndex(Ipre, Iin, Ipost)
            Joffset = CartesianIndex(Ipre, Ioffset, Ipost)
            add_newconnections!(
                tensors, connecting, connections, connection_stati, I, J, Joffset
            )
        end
    end
    return nothing
end

function add_newconnections!(
    tensors::AbstractArray{<:Tensor, N}, connecting,
    connections::Vector{Connection{N}}, 
    connection_stati::Array{Vector{ConnnectionStatus}, N}, 
    I::CartesianIndex{N}, J::CartesianIndex{N}, Joffset::CartesianIndex{N}
) where N
    for (i, leg) in pairs(tensors[I].legs)
        connection_stati[I][i] ≡ uncon || continue
        for space in connecting
            if matching(Outgoing(space), leg)
                j = 0
                for (k, other) in pairs(tensors[J].legs)
                    (connection_stati[J][k] ≡ uncon && dual(leg, other)) || continue
                    @assert j == 0 "multiple legs for connector $(Incoming(space))"
                    j = k
                    connection_stati[I][i] = connection_stati[J][j] = newcon
                end
                push!(connections, Connection(I => Joffset, i => j))
                break
            end
        end
    end
    return nothing
end
function add_newconnections!(tensors, connecting, connections, connection_stati, I)
    add_newconnections!(tensors, connecting, connections, connection_stati, I, I, I)
end

end
