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


include( "Radiosity.jl" )
include( "ExerciseAN2_LGS.jl" )


# setup scene and form factors
# @params: 
#    - resolution: subdivision of the geometry (low, e.g. 8, high, e.g. 32)
#    - emission: light intensity of the area light at the ceiling
# @returns:
#    - Fij NxN matrix of form faktor (N number of faces)
#    - Emission Nx1
function createSceneEssentials( resolution::Int64, emission::Float64 )
    Verts, Faces, FCenters, FNormals, Colors, Emission = setupScene( resolution, emission );
    Fij = systemMatrix( 0.75, Verts, Faces, FCenters, FNormals );
    Fij, Emission, Verts, Faces, Colors
end


vis = prepareCleanVisualizer()


# low res scene
Fij, Emission, Vertices, Faces, Colors = createSceneEssentials( 8, 200.0 );

# or e.g. a high res scene
# Fij, Emission, Vertices, Faces, Colors = createSceneEssentials( 32, 150.0 );

# add solver calls here:
# XGT, RGT = solveGroundTruth( Fij , Emission ) ;
XGT, RGT = jacobi( Fij , Emission , 500 , 1e-8 ) ;

# use radiosity X to scale face colors (invoke multRGB function on all elements)
ColorsShow = map( multRGB, Colors, XGT );

# update 3D scene rendering
showMetaMeshFaceColors( Vertices, Faces, ColorsShow );

# open scene in browser
open(vis)
