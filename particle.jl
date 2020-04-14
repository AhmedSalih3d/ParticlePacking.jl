# VERSION 1.0
# NAIVE IMPLEMENTATION

# Define used libraries
using DelimitedFiles
using LinearAlgebra
using Plots

# GOAL OF THIS CODE:
#   It runs an iterative algorithm on a set of particle positions given an initial
#   velocity field, with the goal of filling white space. More information here;
#   https://www.researchgate.net/publication/256688486_Particle_packing_algorithm_for_SPH_schemes

# Current Benchmark results, using BenchmarkTools, @btime:
# @btime RunAlgorithm()
# 12.940 s (14464000 allocations: 50.04 GiB)
# And using @benchmark:
#@benchmark RunAlgorithm()
#BenchmarkTools.Trial: 
#memory estimate:  50.04 GiB
#allocs estimate:  14464000
#--------------
#minimum time:     12.401 s (3.89% GC)
#median time:      12.401 s (3.89% GC)
#mean time:        12.401 s (3.89% GC)
#maximum time:     12.401 s (3.89% GC)
#--------------
#samples:          1
#evals/sample:     1


# This code consists of:
#   1. drawSquare: 
#           Generate particles in a square shape. The square is not filled.
#   2. read2DParticleGroup: 
#           Reads a particle distribution and hard-coded use of drawSquare
#   3. Some Constants: 
#           These are hardcoded for these exact values
#   4. PackStep: 
#           One step of the algorithm, which moves the particles towards empty space
#   5. RunAlgorithm():
#           Runs the final algorithm

# Function to draw the a square of points
# Inputs are:
#   1. Left corner of square horizontal coordinate, x
#   2. Left corner of square vertical   coordinate, y
#   3. Width       of square                      , s
#   4. Number      of particles on each edge      , n

#---------------------------- drawSquare -----------------------------------------#
function drawSquare(x,y,s,n)
    points = [x   y
              x+s y
              x+s y+s
              x   y+s]
    squarePoints = zeros(4*n,2); 
    
    squarePoints[1:n,1] = range(points[1,1],points[2,1],length=n)
    squarePoints[1:n,2] = range(points[1,2],points[2,2],length=n)
    
    squarePoints[n+1:2*n,1] = range(points[2,1],points[3,1],length=n)
    squarePoints[n+1:2*n,2] = range(points[2,2],points[3,2],length=n)
    
    squarePoints[2*n+1:3*n,1] = range(points[3,1],points[4,1],length=n)
    squarePoints[2*n+1:3*n,2] = range(points[3,2],points[4,2],length=n)
    
    squarePoints[3*n+1:4*n,1] = range(points[4,1],points[1,1],length=n)
    squarePoints[3*n+1:4*n,2] = range(points[4,2],points[1,2],length=n)
    return squarePoints
end

#---------------------------- read2DParticleGroup --------------------------------#
#Function which constructs the relevant initial particle distribution, pg
function read2DParticleGroup(filename::AbstractString)
    coords = readdlm(filename, '\t', Float64, '\n');
    nCoords = size(coords)[1]
    s1 = drawSquare(-1.025,-1.025,2.05,100);
    ns1 = size(s1)[1]

    pg = fill(tuple(0.0,0.0,0.0),nCoords+ns1,1)

    for i = 1:nCoords
        xval = coords[i,1];
        zval = coords[i,2];
        pg[i] = tuple(xval,0,zval)
    end

    for i = 1:ns1
        j = i+nCoords
        xval = s1[i,1];
        zval = s1[i,2];
        pg[j] = tuple(xval,0,zval)
    end

    return pg,nCoords
end
#---------------------------- Some Constants ------------------------------------#
# Set some constants
const H = 0.04
const AD = 348.15;
const FAC = 5/8;
const BETA = 4;
const ZETA = 0.060006;
const V    = 0.0011109;
const DT   = 0.016665;
const H1   = 1/H;

#----------------------------------- PackStep -----------------------------------#
# Put one time step of the algorithm in a function
# Inputs are:
#   1. Initial position of particles, pg
#   2. Initial velocity of particles, updated
#   3. Total   number   of particles, nCoords
function PackStep(pg,u,nCoords)
        # Focus on one particle at at a time, iter
        for iter = 1:nCoords
            # Calculate difference between position of particle iter minus all other particles
            rij = map(x -> pg[iter] .- x, pg)

            # Using LinearAlgebra, calculate the Euclidean distance between iter and each point
            RIJ = norm.(rij)

            # Transform it to a normalized distance
            q   = RIJ ./ H

            # For some reason it becomes a 2D array, I prefer it as 1D..
            q   = dropdims(q,dims=2)

            # Find all particles in near vicinity
            indices = q.<=2;

            # Set the influence of the current particle in focus, iter, to zero
            indices[iter] = 0

            # Extract only the normalized distance from the closest particles to iter
            qc  = q[indices]

            # Calculate main body of the Wendland Kernel derivative, dW/dq
            qq3 =  qc .* (qc .- 2).^3;

            # Calculate the actual dW/dq
            Wq  = AD*FAC*qq3

            # Extract the distance, between, 
            # (x_i[1],x_i[2],x_i[3]) -  (x_j[1],x_j[2],x_j[3]), 
            # which was calculated earlier in "rij"
            x_ij   = first.(rij[indices]);
            z_ij   = last.(rij[indices]);

            # From C++ I have learned to divide before hand.. maybe it doesn't make sense to do so in Julia
            RIJ1 = 1 ./ RIJ[indices];

            # Calculate actual gradient values for particle iter
            Wgx  = sum(Wq .* (x_ij .* RIJ1) * H1);
            Wgz  = sum(Wq .* (z_ij .* RIJ1) * H1);

            # Extract ux and uz components and make it 1D array
            ux   = dropdims(first.(u),dims=2);
            uz   = dropdims(last.(u),dims=2);

            # Calculate change of velocity for particle iter
            dux  = (-BETA .* Wgx * V - ZETA .* ux[iter])*DT;
            duz  = (-BETA .* Wgz * V - ZETA .* uz[iter])*DT;

            # Calculate displacement of particle iter
            dx   = dux*DT;
            dz   = duz*DT;

            # Set new velocity and displacement
            # NOTE: I am aware I "cheat" since I base calculations
            # on updated values over time, so I halfway through
            # would have 50% updated and 50% non-updated values.
            # Since displacements are so small in this method, it
            # really does not do a big difference, if one sticks
            # to the original method
            u[iter]  = u[iter]  .+ tuple(dux,0,duz);
            pg[iter] = pg[iter] .+ tuple(dx,0,dz);
        end
end

#---------------------------- RunAlgorithm ------------------------------------#
function RunAlgorithm()
    # Generate initial data and number of data points
    pg,nCoords = read2DParticleGroup("CircleGridUniform.txt");

    # Generate initial velocity field
    u = map(x -> x.*tuple(1,0.0,-1).*50,pg);
    u[nCoords+1:end] .= tuple.(0.0,0.0,0.0)

    for t = 1:100
        PackStep(pg,u,nCoords)
    end

    # Drop final result
    plot(dropdims(first.(pg),dims=2),dropdims(last.(pg),dims=2), 
        seriestype = :scatter, 
        title = "My Scatter Plot", 
        aspect_ratio=1)
end

# Time and run the main body
@time RunAlgorithm()
