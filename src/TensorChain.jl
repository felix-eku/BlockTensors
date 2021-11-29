module TensorChain

export createchain, connectchain!

using ..BlockTensors
using ..BlockTensors.ConnectionIds


function createchain(
    createtags, tensors::AbstractArray{<:Tensor, N}, repeats::Integer; dim::Integer = N + 1
) where N
    0 ≤ dim ≤ N + 1 || throw(DomainError(dim, "dim not one of 0, ..., N + 1"))
    Base.require_one_based_indexing(tensors)
    if dim == 0
        chain = similar(tensors, repeats, size(tensors)...)
        axis = axes(chain, 1)
        combine_indices = (k, I) -> CartesianIndex(k, I)
    elseif dim == N + 1
        chain = similar(tensors, size(tensors)..., repeats)
        axis = axes(chain, N + 1)
        combine_indices = (k, I) -> CartesianIndex(I, k)
    else
        chain = similar(tensors, setindex(size(tensors), size(tensors, dim) * repeats, dim))
        axis = axes(chain, dim)[begin : size(tensors, dim) : end]
        combine_indices = (k, I) -> setindex(I, I[dim] + k - oneunit(k), dim)
    end
    offset = nextid()
    for k in axis, I in CartesianIndices(tensors)
        t = chain[combine_indices(k, I)] = copy(tensors[I])
        for leg in t.legs
            addtags!(leg; createtags(k, leg.connector)...)
            leg.connection.id == 0 || (leg.connection.id += offset)
        end
    end
    return chain
end
function createchain(createtags, tensor::Tensor, repeats::Integer)
    createchain(createtags, fill(tensor, ()), repeats)
end

function connectchain!(chain::AbstractArray{<:Tensor, N}, connecting; dim = N) where N
    outgoing = Outgoing(connecting)
    incoming = Incoming(connecting)
    lout = lin = 0
    axis = axes(chain, dim)
    for (kout, kin) in zip(@view(axis[begin : end - 1]), @view(axis[begin + 1 : end]))
        lout, lin = connectchain!(
            selectdim(chain, dim, kout), selectdim(chain, dim, kin),
            outgoing, incoming, lout, lin
        )
    end
    return lout, lin
end

function connectchain!(chainout, chainin, connecting)
    outgoing = Outgoing(connecting)
    incoming = Incoming(connecting)
    connectchain!(chainout, chainin, outgoing, incoming)
end
function connectchain!(
    chainout, chainin, outgoing::Outgoing, incoming::Incoming, lout::Integer = 0, lin::Integer = 0
)
    for (tout, tin) in zip(chainout, chainin)
        _, lout, lin = connect!(tout, tin, outgoing, incoming, lout, lin)
    end
    return lout, lin
end

end
