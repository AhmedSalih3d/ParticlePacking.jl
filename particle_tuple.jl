# VERSION 2.0
# NAIVE IMPLEMENTATION

# Define used libraries
using DelimitedFiles
using LinearAlgebra
using Plots
using Base.Threads
using NearestNeighbors

# GOAL OF THIS CODE:
#   It runs an iterative algorithm on a set of particle positions given an initial
#   velocity field, with the goal of filling white space. More information here;
#   https://www.researchgate.net/publication/256688486_Particle_packing_algorithm_for_SPH_schemes

# Current Benchmark results, using BenchmarkTools, @btime:
# @btime RunAlgorithm()
# 6.668953 seconds (24.79 M allocations: 466.441 MiB, 1.34% gc time)
# And using @benchmark:
#@benchmark RunAlgorithm()
#BenchmarkTools.Trial:
#  memory estimate:  496.89 MiB
#  allocs estimate:  25406267
#  --------------
#  minimum time:     6.510 s (0.60% GC)
#  median time:      6.510 s (0.60% GC)
#  mean time:        6.510 s (0.60% GC)
#  maximum time:     6.510 s (0.60% GC)
#  --------------
#  samples:          1
#  evals/sample:     1


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
const A0 = 50
const AD = 348.15;
const FAC = 5/8;
const BETA = 4;
const ZETA = 0.060006;
const V    = 0.0011109;
const DT   = 0.016665;
const H1   = 1/H;
const H2   = 4*H^2;
const R    = 2*H;

#----------------------------------- PackStep -----------------------------------#
# Put one time step of the algorithm in a function
# Inputs are:
#   1. Initial position of particles, pg
#   2. Initial velocity of particles, updated
#   3. Total   number   of particles, nCoords
#   4. The extra amount of particles, nOver
sqnorm(x) = sum(abs2,x)
function PackStep(pg,pg_tmp,u,u_tmp,nCoords,nTot,idxs)
    #idxs = Array{Int64,1};
    @fastmath @inbounds @threads for i = 1:nCoords
        Wgx = 0.0;
        Wgz = 0.0;
        p_i   = pg[i];

        filter!(x->x≠i,idxs[i])

        for j in idxs[i]
            p_j = pg[j];
            rij = p_i .- p_j;
            RIJ = sqnorm(rij);
                RIJ = sqrt(RIJ)
                RIJ1= 1.0 / RIJ;
                q   = RIJ*H1;
                qq3 = q*(q-2)^3;
                Wq  = AD * FAC * qq3;

                x_ij = rij[1];
                z_ij = rij[3];

                Wgx += Wq * (x_ij * RIJ1) * H1;
                Wgz += Wq * (z_ij * RIJ1) * H1;
        end

        # Define here since I use it a lot
        u_i = u[i];

        dux = (-BETA * Wgx * V - ZETA * u_i[1])*DT;
        duz = (-BETA * Wgz * V - ZETA * u_i[3])*DT;

        dx  = dux*DT;
        dz  = duz*DT;

        u_tmp[i]  =   u_i   .+ (dux, 0.0, duz)
        pg_tmp[i] =   pg[i] .+ (dx,  0.0, dz)
    end

    u  .= u_tmp;
    pg .= pg_tmp;
end

#---------------------------- RunAlgorithm ------------------------------------#
function RunAlgorithm()
    # Generate initial data and number of data points
    pg,nCoords = read2DParticleGroup("CircleGridUniform.txt");

    # Generate initial velocity field
    u = map(x -> x.*tuple(1,0.0,-1).*A0,pg);
    u[nCoords+1:end] .= tuple.(0.0,0.0,0.0)

    nOver = size(pg)[1] - nCoords;

    nTot = nCoords + nOver;
    u_tmp  = deepcopy(u)
    pg_tmp = deepcopy(pg)

    #Neighbour search
    pg_arr = reshape(collect(Iterators.flatten(pg)),(3,nTot))
    balltree = BallTree(pg_arr,reorder = false)
    idxs = inrange(balltree, pg_arr, R, false)

    @time for t = 1:100
        if mod(t,8) == 0
            pg_arr .= reshape(collect(Iterators.flatten(pg)),(3,nTot))
            balltree = BallTree(pg_arr,reorder = false)
            idxs .= inrange(balltree, pg_arr, R, false)
        end
        PackStep(pg,pg_tmp,u,u_tmp,nCoords,nTot,idxs)
    end

    # Drop final result
    plot(dropdims(first.(pg),dims=2),dropdims(last.(pg),dims=2),
        seriestype = :scatter,
        title = "My Scatter Plot",
        aspect_ratio=1,
        fmt = :png
        )
    png("ParticleDistribution")
end

# Time and run the main body
RunAlgorithm()
