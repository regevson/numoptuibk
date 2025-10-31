using LinearAlgebra
using Printf


# ------------------ add solver definitions --------------------------
function jacobi(A::Matrix, b::Vector, maxiter = 100, epsilon = 1e-8)
    x = zeros(length(b))                  # example vector init
    @printf "first elem in x %f" x[1]     # example std out ( index starting from 1(!) )

    # ...

end

