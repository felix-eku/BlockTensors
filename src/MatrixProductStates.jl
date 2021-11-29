module MatrixProductStates

export norm, matrixelement, expectationvalue
export bond_dimension
export contractchains
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
    leg = matchlegs(MPS[eachindex(MPS)[bond]], connecting)
    leg.dimensions.totaldim
end

function contractchains(chain1, chain2, auxiliary)
    contractchains(chain1, chain2, auxiliary, auxiliary)
end
function contractchains(chain1, chain2, auxiliary1, auxiliary2)
    contraction = similar(chain1)
    for k in eachindex(chain1, chain2)
        T = chain1[k] * chain2[k]
        incoming = union(
            matching.(Incoming.(auxiliary1), Ref(chain1[k])),
            matching.(Incoming.(auxiliary2), Ref(chain2[k]))
        )
        outgoing = union(
            matching.(Outgoing.(auxiliary1), Ref(chain1[k])),
            matching.(Outgoing.(auxiliary2), Ref(chain2[k]))
        )
        contraction[k] = mergelegs(T, incoming, outgoing)
    end
    return contraction
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
