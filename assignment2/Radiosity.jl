using MeshCat
using ColorTypes
using CoordinateTransformations
using GeometryBasics
using Random
using Statistics
using LinearAlgebra

function patch(O::Point, D1::Vec, D2::Vec, A::Int64, B::Int64)

    V = [ O + D1 * (i-1) + D2 * (j-1)  for j=1:B for i=1:A]

    F = [ NgonFace( 
            i + (j-1) * (A), 
            i + (j-1) * (A) + 1, 
            i + (j-1) * (A) + 1 + A, 
            i + (j-1) * (A) + A )
          for i=1:A-1 for j=1:B-1 ]
   
    V, F
end


function patchDistinct(O::Point, D1::Vec, D2::Vec, A::Int64, B::Int64, FaceOffset::Int64)

    Vtmp = [ [O + D1 * (i-1) + D2 * (j-1),
              O + D1 * (i) + D2 * (j-1),
              O + D1 * (i) + D2 * (j), 
              O + D1 * (i-1) + D2 * (j)]  for j=1:B for i=1:A ]

    faceCenters = mean.(Vtmp)
    faceNormals = fill( normalize( cross( D1, D2) ), size(faceCenters) )
    
    vertices = [(Vtmp...)...]

    faces = [ NgonFace( FaceOffset+i, FaceOffset+i+1, FaceOffset+i+2, FaceOffset+i+3 ) for i=1:4:(A*B*4)-1 ]
    
    vertices, faces, faceCenters, faceNormals
end

function showMetaMeshFaceColorsLambert( Verts, Faces, FColors )
    cols = repeat( FColors, inner=4 );
    geometry = GeometryBasics.Mesh( Verts, Faces )
    mesh_meta = GeometryBasics.meta( geometry, vertexColors=cols )
    material = MeshLambertMaterial(vertexColors=true)
    setobject!(vis[:vertex_color_mesh], mesh_meta, material)
end


function showMetaMeshFaceColors( Verts::Vector{Point3{Float64}}, Faces::Vector{QuadFace{Int64}}, FColors::Vector{RGB{Float64}} )
    cols = repeat( FColors, inner=4 );
    geometry = GeometryBasics.Mesh( Verts, Faces )
    mesh_meta = GeometryBasics.meta( geometry, vertexColors=cols )
    material = MeshBasicMaterial(vertexColors=true)
    setobject!(vis[:vertex_color_mesh], mesh_meta, material)
end


function noisedPatch(Col::RGB, O, D1, D2, A, B, FOffset)
    P, F, C, N = patchDistinct( O, D1, D2, A, B, FOffset )

    cols = fill( Col, size(C) );

    colors = map( 
        function(x) 
            d=1.0 #(0.9 + 0.1*rand())
            RGB(d*x.r, x.g*d, x.b*d)
        end, 
        cols  )

    P, F, C, N, colors
end

function multRGB( a::RGB, b::Float64 )
    RGB( a.r * b, a.g * b, a.b * b);
end

function multRGB( a::RGB, b::RGB )
    RGB( a.r * b.r, a.g * b.g, a.b * b.b);
end

function setupScene( resolution::Int64, emitting::Float64 )

    res = ( resolution ÷ 4 ) * 4

    dX = Vec( 2/res, 0, 0);
    dY = Vec( 0, 2/res, 0 );
    dZ = Vec( 0, 0, 2/res );
    
    cScale = 2.0

    red   = multRGB( RGB(0.3, 0.1, 0.1), cScale )
    green = multRGB( RGB(0.1, 0.3, 0.1), cScale );
    blue  = multRGB( RGB(0.1, 0.1, 0.3), cScale );
    gray  = multRGB( RGB(0.2, 0.2, 0.2), cScale );
    white = multRGB( RGB(1.0, 1.0, 1.0), cScale );

    # V  ... vertices
    # F  ... faces
    # FC ... face centers
    # C  ... face colors

    # light
    V1, F1, FC1, FN1, C1 = noisedPatch( white,  Point( 0.2, -0.2, 0.98 ), -dX*0.8, dY*0.8, res÷4, res÷4, 0 );

    # walls # (-1, -1, -1) -- (1, 1, 1)
    V2, F2, FC2, FN2, C2 = noisedPatch( red,   Point(-1, -1, 1 ), dX, -dZ, res, res, last(F1)[3]+1 );
    V3, F3, FC3, FN3, C3 = noisedPatch( green, Point(-1,  1, -1 ), dX, dZ, res, res, last(F2)[3]+1  );
    V4, F4, FC4, FN4, C4 = noisedPatch( blue,  Point(-1, -1, -1 ), dY, dZ, res, res, last(F3)[3]+1  );
    V5, F5, FC5, FN5, C5 = noisedPatch( gray,  Point(1, -1,  1 ), -dX, dY, res, res, last(F4)[3]+1  );
    V6, F6, FC6, FN6, C6 = noisedPatch( gray,  Point(-1, -1, -1 ), dX, dY, res, res, last(F5)[3]+1 );    

    #box # (-0.6, -0.6, -0.9) -- (0, 0, 0)
    V7, F7, FC7, FN7, C7 = noisedPatch( multRGB( gray, 1.2 ) , Point( 0, -0.6, -0.9 ), -dX*1.2, dY*1.2, res÷4, res÷4, last(F6)[3]+1 );
    V8, F8, FC8, FN8, C8 = noisedPatch( multRGB( gray, 1.3 ) , Point(-0.6, -0.6, -0.9 ), dX*1.2, dZ*1.8, res÷4, res÷4, last(F7)[3]+1 );
    V9, F9, FC9, FN9, C9 = noisedPatch( multRGB( gray, 1.4 ) , Point(-0.6, -0.6, 0.0 ),  dX*1.2, dY*1.2, res÷4, res÷4, last(F8)[3]+1 );
    VA, FA, FCA, FNA, CA = noisedPatch( multRGB( gray, 1.2 ) , Point(-0.6,  0.0, 0.0 ), dX*1.2, -dZ*1.8, res÷4, res÷4, last(F9)[3]+1 );
    VB, FB, FCB, FNB, CB = noisedPatch( multRGB( gray, 1.2 ) , Point(-0.6, -0.6, -0.9 ), dZ*1.8, dY*1.2, res÷4, res÷4, last(FA)[3]+1 );
    VC, FC, FCC, FNC, CC = noisedPatch( multRGB( gray, 1.2 ) , Point( 0.0, 0.0, -0.9 ), dZ*1.8, -dY*1.2, res÷4, res÷4, last(FB)[3]+1 );

    # flatten and concat arrays 
    V  = [ V1...,  V2...,  V3...,  V4...,  V5...,  V6...,  V7... , V8... , V9... , VA... , VB... , VC... ];
    F  = [ F1...,  F2...,  F3...,  F4...,  F5...,  F6...,  F7... , F8... , F9... , FA... , FB... , FC... ];
    FC = [ FC1..., FC2..., FC3..., FC4..., FC5..., FC6..., FC7..., FC8..., FC9..., FCA..., FCB..., FCC...];
    FN = [ FN1..., FN2..., FN3..., FN4..., FN5..., FN6..., FN7..., FN8..., FN9..., FNA..., FNB..., FNC...];
    C  = [ C1...,  C2...,  C3...,  C4...,  C5...,  C6...,  C7... , C8... , C9... , CA... , CB... , CC... ];
    
    E = zeros( size(FC)[1] )
    E[ 1 : size(FC1)[1]] .= emitting
            
    V, F, FC, FN, C, E
end

function prepareCleanVisualizer()
    vis = Visualizer()
    setprop!(vis["/Background"], "visible", true )
    setprop!(vis["/Grid"], "visible", false )
    setprop!(vis["/Axes"], "visible", false )

    vis
end

# PO: origin of the quad (parallelograms only)
# PU: direction and length of edge 1
# PV: direction and length of edge 2
function rayQuadIntersect( RO::Point, RD::Vec, PO::Point, PU::Vec, PV::Vec )
    eps = 0.0001

    rd = normalize( RD )

    pc  = cross( rd, PV )
    det = dot( PU, pc )

    if( det > -eps && det < eps)
        return false, Inf64, Point(0.0, 0.0, 0.0)
    else
        detInv = 1.0 / det;

        # distance from plane to ray origins
        rp = RO - PO
        u = dot( rp, pc) * detInv;

        if( u < 0.0 || u > 1.0 )
            return false, Inf64, Point(0.0, 0.0, 0.0)
        else
            qc = cross( rp, PU )
            v = dot(rd, qc ) * detInv;

            if(v < 0.0 || v > 1.0)
                return false, Inf64, Point(0.0, 0.0, 0.0)
            else
                rt = dot(PV, qc ) * detInv;
                intPosition = RO + rt * rd;
                return true, rt, intPosition
            end
        end
    end
end

function hitBox( O::Point, D::Vec )
    hits = [rayQuadIntersect(O, D, Point( -0.6, -0.6, -0.9 ), Vec(0, 0.6, 0), Vec(0.6, 0, 0)),  # bottom
            rayQuadIntersect(O, D, Point( -0.6, -0.6,  0.0 ), Vec(0, 0.6, 0), Vec(0.6, 0, 0)),  # top
            rayQuadIntersect(O, D, Point( -0.6, -0.6, -0.9 ), Vec(0, 0.6, 0), Vec(0, 0, 0.9)),  # back
            rayQuadIntersect(O, D, Point(  0.0, -0.6, -0.9 ), Vec(0, 0.6, 0), Vec(0, 0, 0.9)),  # front
            rayQuadIntersect(O, D, Point( -0.6, -0.6, -0.9 ), Vec(0, 0, 0.9), Vec(0.6, 0, 0)),  # left
            rayQuadIntersect(O, D, Point( -0.6, -0.0, -0.9 ), Vec(0, 0, 0.9), Vec(0.6, 0, 0)) ] # right

    success = filter( x -> x[1], hits  )
    if( size(success)[1] == 0 )
        false, Point(0.0, 0.0, 0.0)
    else
        closest = sort( success, by = x -> x[2] )

        true, closest[1][3] 
    end
end

function inBox( P::Point, Min::Point, Max::Point )
    if( P[1] <= Max[1] && P[2] <= Max[2] && P[3] <= Max[3] && 
        P[1] >= Min[1] && P[2] >= Min[2] && P[3] >= Min[3] )
        true
    else
        false
    end
end    

function distance2( A::Point, B::Point )
    V = A - B
    dot(V, V)
end


function FPerPatchJ( Pi::Point, Ni::Vec, Pj::Point, Nj::Vec, Aj::Float64 )
    AB = Pj - Pi
    r2 = dot(AB, AB)
    ABN = normalize( AB )
    cosPhi_i = dot( Ni, ABN )
    cosPhi_j = dot( Nj, -ABN )

    F = cosPhi_i * cosPhi_j * Aj / ( pi * r2 )
    if F < 0.0 
        0.0
    else
        F
    end
end

function areaPatch( i::Int64, Verts, Faces )
    A = Verts[Faces[i][1]]
    B = Verts[Faces[i][2]]
    C = Verts[Faces[i][3]]

    AB = B - A
    BC = C - B

    norm( cross(AB, BC) )
end

function FPerPatchMC( i::Int64, Verts, Faces, FaceCenters, FaceNormals, fraction::Float64 )
    nrFaces = size(Faces)[1]
    iterations = floor( Int64, nrFaces * fraction )

    Fij = 0.0
    for j = 1:iterations
        j = rand(1:nrFaces)

        D = Vec(normalize( FaceCenters[j] - FaceCenters[i]  ))
        if( hitBox( FaceCenters[i], D )[1] == false)
            Aj = areaPatch( j, Verts, Faces )
            Fij += FPerPatchJ( FaceCenters[i], FaceNormals[i], FaceCenters[j], FaceNormals[j], Aj )
        end
    end

    Ai = areaPatch( i, Verts, Faces )
    Fij /= iterations
    Fij / Ai
end

function FPerPatch( i::Int64, j::Int64, Verts, Faces, FaceCenters, FaceNormals )
    Fij = 0.0
    D = Vec(normalize( FaceCenters[j] - FaceCenters[i]  ))
    if( hitBox( FaceCenters[i], D )[1] == false || 
        inBox(FaceCenters[i], Point(-0.6, -0.6, -0.9), Point(0.0, 0.0, 0.0) ) )
        Aj = areaPatch( j, Verts, Faces )
        Fij += FPerPatchJ( FaceCenters[i], FaceNormals[i], FaceCenters[j], FaceNormals[j], Aj )
    end

    Ai = areaPatch( i, Verts, Faces )
    Fij / Ai
end

function getFij( Verts, Faces, FaceCenters, FaceNormals )
    s = size(FaceCenters)[1]
    A = zeros( s, s )
    for i in 1:s 
        for j in 1:s
            if i != j
                A[i,j] = FPerPatch( i, j, Verts, Faces, FaceCenters, FaceNormals )
            end
        end
    end
    A
end
 
# ! to mark passing back by reference (Fij)
function symetrifyFij!( Fij, Verts, Faces )
    s = size(Faces)[1]
    for i in 1:s 
        for j in 1:s
            if i > j 
                Ai = areaPatch(i, Verts, Faces)
                Aj = areaPatch(j, Verts, Faces)
                AiFij = Ai * Fij[i, j]
                AjFji = Aj * Fij[j, i]
                Avg = (AiFij + AjFji) / 2.0
                Fij[i, j] = Avg / Ai
                Fij[j, i] = Avg / Aj
            end
        end
    end
end


function systemMatrix( albedo::Float64, Verts, Faces, FaceCenters, FaceNormals )
    Fij = getFij( Verts, Faces, FaceCenters, FaceNormals);
    symetrifyFij!( Fij, Verts, Faces );
    Fi = sum(Fij, dims=2); 
    FiMax = maximum( Fi );
    Fij = Fij ./ FiMax;
    Fij2 = (-albedo .* Fij) + 1.0*Matrix(I, size(Fij)[1], size(Fij)[1]);
    Fij2
end


# debugging helper functions
function lightByNormalDirection!( FaceColors, FaceNormals, N::Vec )
    for i = 1:size(FaceNormals)[1]
        d = max( 0.0, dot( N, FaceNormals[i] ) )
        FaceColors[i] = multRGB( FaceColors[i], d  )       
    end
end 

function directShadowCast!(FaceColors, FaceCenters)
    for i=1:size(FaceCenters)[1]
        lightPos = Point(0.55, 0.5, 0.9)

        dir = Vec( lightPos - FaceCenters[i] )
    
        hit, point = hitBox(FaceCenters[i], dir)

        if( hit && distance2( FaceCenters[i], lightPos )  >= distance2( point, lightPos ) )
            FaceColors[i] = multRGB(FaceColors[i], 0.65)
        end
    end
end

function getFormFactorSumPerFace(Vertices, Faces, FaceCenters, FaceNormals, Colors)
    Fij = getFij( Vertices, Faces, FaceCentersC, FaceNormals)
    symetrifyFij!( Fij, Vertices, F )
    Fi = sum(Fij, dims=2) 
    FiMax = maximum( Fi )
    Fij = Fij ./ FiMax
    Fi = Fi ./ FiMax
    C1 = map( multRGB, Colors, Fi );
    C1
end