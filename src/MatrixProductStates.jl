module MatrixProductStates

export norm, matrixelement, expectationvalue
export MPO_MPS_contraction
export exchangegauge, canonicalize!, bond_canonicalize!

using ..BlockTensors
using ..TensorChain: findconnections

function adjointMPS(MPS)
    connections, _stati = findconnections(MPS, ())
    adjoint_MPS = adjoint.(MPS)
    for connection in connections
        I, J = connection.positions; i, j = connection.legs
        connect!(adjoint_MPS[I].legs[i], adjoint_MPS[J].legs[j])
    end
    return adjoint_MPS
end

function norm(MPS)
    adjoint_MPS = adjointMPS(MPS)
    contraction = 1
    for k in eachindex(MPS, adjoint_MPS)
        contraction = (contraction * MPS[k]) * adjoint_MPS[k]
    end
    return √abs(contraction)
end

function matrixelement(braMPS, MPO, ketMPS)
    contraction = 1
    for k in eachindex(braMPS, MPO, ketMPS)
        for leg in Iterators.flatten((ketMPS[k].legs, braMPS[k].legs))
            duals = filter(MPOleg -> dual(leg, MPOleg), MPO[k].legs)
            isempty(duals) && continue
            any(connected(leg, connectable) for connectable in duals) && continue
            connect!(leg, only(duals))
        end
        contraction = ((contraction * braMPS[k]) * MPO[k]) * ketMPS[k]
    end
    return contraction
end

expectationvalue(MPO, ketMPS) = matrixelement(adjointMPS(ketMPS), MPO, ketMPS)


function MPO_MPS_contraction(MPO, MPS, MPOmerging, MPSmerging)
    contractionMPS = similar(MPS)
    for k in eachindex(MPO, MPS)
        contraction = MPO[k] * MPS[k]
        incoming = union(
            matching.(Incoming.(MPSmerging), Ref(MPS[k])),
            matching.(Incoming.(MPOmerging), Ref(MPO[k]))
        )
        outgoing = union(
            matching.(Outgoing.(MPSmerging), Ref(MPS[k])),
            matching.(Outgoing.(MPOmerging), Ref(MPO[k]))
        )
        contractionMPS[k] = mergelegs(contraction, incoming, outgoing)
    end
    return contractionMPS
end
function MPO_MPS_contraction(MPO, MPS, merging)
    MPO_MPS_contraction(MPO, MPS, merging, merging)
end

function exchangegauge(A::Tensor, B::Tensor, connecting)
    Q, R = qr(A, connecting...)
    return Q, R * B
end

function exchangegauge(A::Tensor, B::Tensor, connecting, maxblockdim)
    U, S, V = svd(A, connecting...; maxblockdim)
    return U, (S * V) * B
end

function canonicalize!(MPS, K, connecting, params...; normalize::Bool = true)
    for (k1, k2) in zip(K[begin : end - 1], K[begin + 1 : end])
        MPS[k1], MPS[k2] = exchangegauge(MPS[k1], MPS[k2], connecting, params...)
    end
    if normalize
        norm = √(MPS[K[end]]'MPS[K[end]])
        MPS[K[end]] = MPS[K[end]] / norm
        return norm
    end
    return nothing
end

function bond_canonicalize!(MPS, bond, connecting)
    K = eachindex(MPS)
    knext = K[bond + oneunit(bond)]
    canonicalize!(MPS, K[begin : bond], connecting, normalize = false)
    canonicalize!(MPS, K[bond : end], dual(connecting), normalize = false)
    U, S, V = svd(MPS[K[bond]], connecting)
    MPS[K[bond]] = U * S
    MPS[knext] = V * MPS[knext]
    return S
end

end
