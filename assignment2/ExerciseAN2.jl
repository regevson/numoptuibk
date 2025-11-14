#######################################################
# OptiNum Assignment N.2
# Application example: radiosity of a cornell box scene
# Compare different linear system solvers
# IGS 9.11.2021
# Marcel Ritter, Nikolaus Rauch
#######################################################


# before execution setup project path
#cd("c:\\2021\\Teaching\\OptiNum\\PS\\Code\\Assignment_2")


using Pkg
function installPackages()
    Pkg.add("MeshCat")
    Pkg.add("ColorTypes")
    Pkg.add("CoordinateTransformations")
    Pkg.add("GeometryBasics")
    Pkg.add("Random")
    Pkg.add("Statistics")
    Pkg.add("LinearAlgebra")
end

# installPackages() # <-- comment line after first execution, or the whole Pkg block above


include("Radiosity.jl")
include("ExerciseAN2_LGS.jl")


# setup scene and form factors
# @params: 
#    - resolution: subdivision of the geometry (low, e.g. 8, high, e.g. 32)
#    - emission: light intensity of the area light at the ceiling
# @returns:
#    - Fij NxN matrix of form faktor (N number of faces)
#    - Emission Nx1
function createSceneEssentials(resolution::Int64, emission::Float64)
    Verts, Faces, FCenters, FNormals, Colors, Emission = setupScene(resolution, emission)
    Fij = systemMatrix(0.75, Verts, Faces, FCenters, FNormals)
    Fij, Emission, Verts, Faces, Colors
end


vis = prepareCleanVisualizer()


# low res scene
Fij, Emission, Vertices, Faces, Colors = createSceneEssentials(8, 200.0);

# or e.g. a high res scene
# Fij, Emission, Vertices, Faces, Colors = createSceneEssentials( 32, 150.0 );

# add solver calls here:

using Plots
gr()

plt = plot(
    title="Residuals of different solvers",
    xlabel="Iteration",
    ylabel="Residual Norm",
    yscale=:log10,
)

epsilon = 1e-10
maxiter = 500

# number of repetitions used for timing each solver
n_runs = 20

# ---------- Residual plot (single run per solver) ----------

# Ground truth
# XGT, RGT = solveGroundTruth(Fij, Emission)
# RGT is scalar residual -> horizontal line
# hline!(plt, [RGT], label="Ground Truth Residual", linestyle=:dash)

# Jacobi
XJ, RJ = jacobi(Fij, Emission, maxiter, epsilon)
plot!(plt, RJ, label="Jacobi")

# Gauss–Seidel
XGS, RGS = gaussSeidel(Fij, Emission, maxiter, epsilon)
plot!(plt, RGS, label="Gauss-Seidel")

# SOR, ω = 1.25
XSOR1, RSOR1 = sor(Fij, Emission, maxiter, epsilon, 1.25)
plot!(plt, RSOR1, label="SOR, ω=1.25")

# SOR, ω = 1.8
XSOR2, RSOR2 = sor(Fij, Emission, maxiter, epsilon, 1.8)
plot!(plt, RSOR2, label="SOR, ω=1.8")

# Gradient Descent
XGD, RGD = gradientDescent(Fij, Emission, maxiter, epsilon)
plot!(plt, RGD, label="Gradient Descent")

# Conjugate Gradient
XCG, RCG = conjugateGradient(Fij, Emission, maxiter, epsilon)
plot!(plt, RCG, label="Conjugate Gradient")

display(plt)
png(plt, "residuals_comparison.png")

# ============================
#   Timing with loops + confidence intervals
# ============================

using Statistics

methods = [
    "Ground Truth",
    "Jacobi",
    "Gauss-Seidel",
    "SOR (ω=1.25)",
    # "SOR (ω=1.8)",
    "Gradient Descent",
    "Conjugate Gradient",
]

# store all time measurements for each method
times = Dict{String,Vector{Float64}}(m => Float64[] for m in methods)

for run in 1:n_runs
    # Ground truth
    t = @elapsed solveGroundTruth(Fij, Emission)
    push!(times["Ground Truth"], t)

    # Jacobi
    t = @elapsed jacobi(Fij, Emission, maxiter, epsilon)
    push!(times["Jacobi"], t)

    # Gauss–Seidel
    t = @elapsed gaussSeidel(Fij, Emission, maxiter, epsilon)
    push!(times["Gauss-Seidel"], t)

    # SOR, ω = 1.25
    t = @elapsed sor(Fij, Emission, maxiter, epsilon, 1.25)
    push!(times["SOR (ω=1.25)"], t)

    # Gradient Descent
    t = @elapsed gradientDescent(Fij, Emission, maxiter, epsilon)
    push!(times["Gradient Descent"], t)

    # Conjugate Gradient
    t = @elapsed conjugateGradient(Fij, Emission, maxiter, epsilon)
    push!(times["Conjugate Gradient"], t)
end

# compute mean times and 95% confidence intervals
means = Float64[]
errs = Float64[]  # half-width of CI

for m in methods
    ts = times[m]
    mu = mean(ts)
    sigma = std(ts)
    ci_halfwidth = 1.96 * sigma / sqrt(length(ts))  # 95% CI
    push!(means, mu)
    push!(errs, ci_halfwidth)
end

plt_time = bar(
    methods,
    means,
    yerr=errs,
    title="Mean ± 95% CI, n = $n_runs",
    xlabel="Method",
    ylabel="Time [s]",
    legend=false,
    xticks=(1:length(methods), methods),
    xrotation=45,
)

display(plt_time)
png(plt_time, "solver_runtimes.png")

# use radiosity X to scale face colors (invoke multRGB function on all elements)
ColorsShow = map(multRGB, Colors, XCG);

# update 3D scene rendering
showMetaMeshFaceColors(Vertices, Faces, ColorsShow);

# open scene in browser
open(vis)
