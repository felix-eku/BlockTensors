module MatrixProductStates

export norm, matrixelement, expectationvalue
export bond_dimension
export MPO_MPS_contraction
export exchangegauge, canonicalize!, bond_canonicalize!
export density_probabilities, entropy, entanglement_entropy

using ..BlockTensors
using ..BlockTensors.ConnectionIds
using ..BlockTensors.TensorChain

function adjointMPS(MPS, physical)
    aMPS = adjoint.(MPS)
    offset = nextid()
    for T in aMPS, leg in T.legs
        matching(physical, leg) || (leg.connection.id += offset)
    end
    return aMPS
end

function norm(MPS, physical)
    contraction = 1
    aMPS = adjointMPS(MPS, physical)
    connectchain!(MPS, aMPS, physical)
    for (T, aT) in zip(MPS, aMPS)
        contraction = (contraction * T) * aT
    end
    return √abs(contraction)
end

function matrixelement(braMPS, MPO, ketMPS)
    contraction = 1
    for (braT, opT, ketT) in zip(braMPS, MPO, ketMPS)
        contraction = ((contraction * braT) * opT) * ketT
    end
    return contraction
end
function matrixelement(braMPS, MPO, ketMPS, physical)
    connectchain!(ketMPS, MPO, physical)
    connectchain!(MPO, braMPS, physical)
    matrixelement(braMPS, MPO, ketMPS)
end

function expectationvalue(MPO, ketMPS, physical)
    matrixelement(adjointMPS(ketMPS, physical), MPO, ketMPS, physical)
end


function bond_dimension(MPS, bond, connecting)
    only(matching(connecting, MPS[eachindex(MPS)[bond]])).dimensions.totaldim
end


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


function exchangegauge(A::Tensor, B::Tensor, connecting; truncation...)
    if isempty(truncation)
        Q, R = qr(A, connecting...)
        return Q, R * B
    else
        U, S, V = svd(A, connecting...; truncation...)
        return U, (S * V) * B
    end
end

function canonicalize!(MPS, K, connecting; normalize::Bool = true, truncation...)
    for (k1, k2) in zip(K[begin : end - 1], K[begin + 1 : end])
        MPS[k1], MPS[k2] = exchangegauge(MPS[k1], MPS[k2], connecting; truncation...)
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
    canonicalize!(MPS, reverse(K[bond : end]), dual(connecting), normalize = false)
    U, S, V, singulars = svd(MPS[K[bond]], connecting)
    MPS[K[bond]] = U * S
    MPS[knext] = V * MPS[knext]
    return singulars
end

function density_probabilities(singulars)
    probs = singulars.^2
    return probs / sum(probs)
end
function density_probabilities(MPS, bond, connecting)
    density_probabilities(bond_canonicalize!(MPS, bond, connecting))
end

entropy(p) = - sum(p .* log.(p))
entanglement_entropy = entropy ∘ density_probabilities


end
