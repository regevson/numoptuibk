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
        residual_norm = norm(r)
        push!(residuals, residual_norm)

        if residual_norm < epsilon
            return x_new, residuals
        end

        x = x_new
    end

    return x_new, residuals
end

function gaussSeidel(A::Matrix, b::Vector, maxiter=100, epsilon=1e-8)
    x = zeros(length(b))
    x_new = copy(x)
    residuals = Float64[]

    # Compute Matrices once (independent of x)
    D = diagm(0 => diag(A))
    DL = LowerTriangular(A)
    U = UpperTriangular(A) - D

    for iter = 1:maxiter
        x_new = DL \ (b - U * x)

        # residuals
        r = b - A * x_new
        residual_norm = norm(r)
        push!(residuals, residual_norm)

        if residual_norm < epsilon
            return x_new, residuals
        end

        x = x_new
    end

    return x_new, residuals
end

function sor(A::Matrix, b::Vector, maxiter=100, epsilon=1e-8, omega::Float64=1.25)
    # omega is the relaxation factor
    x = zeros(length(b))
    x_new = copy(x)
    residuals = Float64[]

    # Compute Matrices once (independent of x)
    D = diagm(0 => diag(A))
    L = LowerTriangular(A) - D
    DwL = D + omega * L
    U = UpperTriangular(A) - D

    for iter = 1:maxiter
        x_new = DwL \ (omega * b - (omega * U + (omega - 1) * D) * x)

        # residuals
        r = b - A * x_new
        residual_norm = norm(r)
        push!(residuals, residual_norm)

        if residual_norm < epsilon
            return x_new, residuals
        end

        x = x_new
    end

    return x_new, residuals
end

function gradientDescent(A::Matrix, b::Vector, maxiter=100, epsilon=1e-8)
    # Use Steepest Descent Method
    x = zeros(eltype(A), length(b))
    r = b - A * x
    residuals = Float64[norm(r)]

    for iter = 1:maxiter
        # Do Maxtirx x Vector only once and store
        Ar = A * r

        # alpha = (r^T r) / (r^T A r)
        alpha = dot(r, r) / dot(r, Ar)
        # x_k+1 = x_k + alpha * r_k
        x .+= alpha .* r # "." for allocation avoidance
        # r_k+1 = r - alpha * A * r_k
        r .-= alpha .* Ar

        residual_norm = norm(r)
        push!(residuals, residual_norm)

        if residual_norm < epsilon
            return x, residuals
        end
    end

    return x, residuals
end

function conjugateGradient(A::Matrix, b::Vector, maxiter=100, epsilon=1e-8)
    x = zeros(eltype(A), length(b))
    r = b - A * x
    p = copy(r)
    residuals = Float64[norm(r)]

    for iter = 1:maxiter
        Ap = A * p
        rr = dot(r, r)
        alpha = rr / dot(p, Ap)

        # x_k+1 = x_k + alpha * p_k
        x += alpha * p
        # r_k+1 = r_k - alpha * A * p_k
        r_new = r .- alpha .* Ap

        residual_norm = norm(r_new)
        push!(residuals, residual_norm)

        if residual_norm < epsilon
            break
        end

        # we never need beta^(k), just beta^(k+1) so we do not need to update at beginning of iteration
        beta_new = dot(r_new, r_new) / rr

        p .= r_new + beta_new * p
        r .= r_new
    end

    return x, residuals
end

function solveGroundTruth( A::Matrix, b::Vector )
    x = A \ b
    r = norm( A * x - b ) / norm(b)
    return x, r
end

