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
        x_new = DL \ (b - U*x)

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
        push!(residuals, norm(r))

        if norm(r) < epsilon
            return x_new, residuals
        end

        x = x_new
    end

    return x_new, residuals
end

function gradientDescent(A::Matrix, b::Vector, maxiter=100, epsilon=1e-8)
    # Use Steepest Descent Method
    x = zeros(length(b))
    x_new = copy(x)
    residuals = Float64[]

    r = 0
    r_new = 0

    for iter = 1:maxiter
        # r^(k+1) from last iteration is now r^(k).
        r = r_new

        if (iter == 1)
            r = b - A * x
        end

        # Do Maxtirx x Vector only once and store
        Ar = A * r
        
        alpha = (transpose(r) * r) / (transpose(r) * Ar)
        x_new = x + alpha * r
        r_new = r - alpha * Ar
        push!(residuals, norm(r_new))

        if norm(r) < epsilon
            return x_new, residuals
        end

        x = x_new
    end

    return x_new, residuals
end

function conjugateGradient(A::Matrix, b::Vector, maxiter=100, epsilon=1e-8)
    x = zeros(length(b))
    x_new = copy(x)
    residuals = Float64[]

    
    r = 0
    r_new = 0
    p = 0
    p_new = 0

    for iter = 1:maxiter
        # r^(k+1) from last iteration is now r^(k). The same is true for p.
        r = r_new
        p = p_new

        if (iter == 1)
            r = b - A * x
            p = r
        end
        
        alpha = (transpose(r)*r) / (transpose(p) * A * p)
        x_new = x + alpha * p
        r_new = r - alpha * A * p
        push!(residuals, norm(r_new))

        if norm(r) < epsilon
            return x_new, residuals
        end

        # we never need beta^(k), just beta^(k+1) so we do not need to update at beginning of iteration
        beta_new = (transpose(r_new) * r_new) / (transpose(r) * r)
        p_new = r_new + beta_new * p

        x = x_new
    end

    return x_new, residuals
end

function solveGroundTruth( A::Matrix, b::Vector )
    x = A \ b
    r = norm( A * x - b ) / norm(b)
    return x, r
end

