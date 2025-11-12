using LinearAlgebra
using Printf


# ------------------ add solver definitions --------------------------
function jacobi(A::Matrix, b::Vector, maxiter=100, epsilon=1e-8)
    x = zeros(length(b))
    x_new = copy(x)
    residuals = Float64[]

    # Compute D and R matrices once (independent of x)
    D = diagm(0 => diag(A))
    R = A - D
    Dinv = Diagonal(1.0 ./ diag(A))

    for iter = 1:maxiter
        x_new = Dinv * (b - R * x)

        # residuals
        r = b - A * x_new
        push!(residuals, norm(r))

        if norm(r) < epsilon
            return x_new, residuals
        end

        x = x_new
    end

    return x_new, residuals
end

function solveGroundTruth( A::Matrix, b::Vector )
    x = A \ b
    r = norm( A * x - b ) / norm(b)
    return x, r
end

