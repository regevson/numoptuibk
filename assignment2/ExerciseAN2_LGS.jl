using LinearAlgebra
using Printf


# ------------------ add solver definitions --------------------------
function jacobi(A::Matrix, b::Vector, maxiter=100, epsilon=1e-8)
    x = zeros(length(b))
    residuals = Float64[]

    # Matrix notation: x_k+1 = D^-1 * (b - (A - D) * x_k)
    D = Diagonal(A)
    Dinv = inv(D)
    AD = A - D

    for iter = 1:maxiter
        x_new = Dinv * (b - AD * x)

        # residuals
        r = b - A * x_new
        residual_norm = norm(r)
        push!(residuals, residual_norm)

        if residual_norm < epsilon
            return x_new, residuals
        end

        x = x_new
    end

    return x, residuals
end

function gaussSeidel(A::Matrix, b::Vector, maxiter=100, epsilon=1e-8)
    x = zeros(length(b))
    residuals = Float64[]

    # Matrix decomposition: A = D + L + U
    D = Diagonal(A)
    L = LowerTriangular(A) - D
    U = UpperTriangular(A) - D
    
    # Precompute (D+L) for the solver
    DL = D + L

    for iter = 1:maxiter
        # Iteration rule: x_k+1 = (D+L)^-1 * (b - U * x_k)
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

    return x, residuals
end

function sor(A::Matrix, b::Vector, maxiter=100, epsilon=1e-8, omega::Float64=1.25)
    # omega is the relaxation factor
    x = zeros(length(b))
    residuals = Float64[]

    # Matrix decomposition: A = D + L + U
    D = Diagonal(A)
    L = LowerTriangular(A) - D
    U = UpperTriangular(A) - D

    # Precompute matrices for iteration
    # x_k+1 = (D + ωL)^-1 * (ωb - (ωU + (ω-1)D)x_k)
    DwL = D + omega * L
    N = omega * U + (omega - 1) * D

    for iter = 1:maxiter
        x_new = DwL \ (omega * b - N * x)

        # residuals
        r = b - A * x_new
        residual_norm = norm(r)
        push!(residuals, residual_norm)

        if residual_norm < epsilon
            return x_new, residuals
        end

        x = x_new
    end

    return x, residuals
end

function gradientDescent(A::Matrix, b::Vector, maxiter=100, epsilon=1e-8)
    # Use Steepest Descent Method
    # Initialization with x^(0)
    x = zeros(eltype(A), length(b))
    r = b - A * x
    residuals = Float64[norm(r)]

    for iter = 1:maxiter
        # alpha_k = (r_k^T r_k) / (r_k^T A r_k)
        Ar = A * r
        rr = dot(r, r)
        alpha = rr / dot(r, Ar)
        
        # x_k+1 = x_k + alpha_k * r_k
        x .+= alpha .* r
        
        # r_k+1 = r_k - alpha_k * A * r_k
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
    # Initialization step, given x^(0)
    x = zeros(eltype(A), length(b))
    r = b - A * x
    p = copy(r)
    residuals = Float64[norm(r)]

    for iter = 1:maxiter
        # Iterative update rule
        # alpha_k = (r_k^T r_k) / (p_k^T A p_k)
        Ap = A * p
        rr = dot(r, r)
        alpha = rr / dot(p, Ap)

        # x_k+1 = x_k + alpha_k p_k
        x += alpha * p
        # r_k+1 = r_k - alpha_k * A * p_k
        r_new = r .- alpha .* Ap
        
        residual_norm = norm(r_new)
        push!(residuals, residual_norm)

        if residual_norm < epsilon
            return x, residuals
        end

        # beta_k+1 = (r_k+1^T r_k+1) / (r_k^T r_k)
        beta = dot(r_new, r_new) / rr

        # p_k+1 = r_k+1 + beta_k+1 p_k
        p .= r_new + beta * p
        
        # Update r for next iteration
        r .= r_new
    end

    return x, residuals
end

function solveGroundTruth(A::Matrix, b::Vector)
    x = A \ b
    r = norm(A * x - b)
    return x, r
end

