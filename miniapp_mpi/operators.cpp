// TODO : halo exchange

//******************************************
// operators.cpp
// based on min-app code written by Oliver Fuhrer, MeteoSwiss
// modified by Ben Cumming, CSCS
// *****************************************

// Description: Contains simple operators which can be used on 3d-meshes

#include <iostream>

#include <mpi.h>

#include "data.h"
#include "operators.h"
#include "stats.h"

#define N 0
#define B 1
#define W 2
#define E 3

int tag_from(int x){
    return (x^1);
}

namespace operators {

void diffusion(const data::Field &U, data::Field &S)
{
    using data::options;
    using data::domain;

    using data::bndE;
    using data::bndW;
    using data::bndN;
    using data::bndS;

    using data::buffE;
    using data::buffW;
    using data::buffN;
    using data::buffS;

    using data::x_old;

    using data::comm_cart;

    double dxs = 1000. * (options.dx * options.dx);
    double alpha = options.alpha;
    int nx = domain.nx;
    int ny = domain.ny;
    int iend  = nx - 1;
    int jend  = ny - 1;

    MPI_Request sndN_req, rcvN_req, sndS_req, rcvS_req, sndE_req, rcvE_req, sndW_req, rcvW_req;
    if (domain.neighbour_north >= 0)
    {
        for(int i=0;i<=iend;i++)
            buffN[i] = U(i, jend);
        MPI_Irecv(&bndN[0], nx, MPI_DOUBLE, domain.neighbour_north, tag_from(N), comm_cart, &rcvN_req);
        MPI_Isend(&buffN[0], nx, MPI_DOUBLE, domain.neighbour_north, N, comm_cart, &sndN_req);
    }

    if (domain.neighbour_south >= 0)
    {
        for(int i=0;i<=iend;i++)
            buffS[i] = U(i, 0);
        MPI_Irecv(&bndS[0], nx, MPI_DOUBLE, domain.neighbour_south, tag_from(B), comm_cart, &rcvS_req);
        MPI_Isend(&buffS[0], nx, MPI_DOUBLE, domain.neighbour_south, B, comm_cart, &sndS_req);
    }

    if (domain.neighbour_east >= 0)
    {
        for (int j = 0; j <=jend; j++)
            buffE[j] = U(0, j);
        MPI_Irecv(&bndE[0], ny, MPI_DOUBLE, domain.neighbour_east, tag_from(E), comm_cart, &rcvE_req);
        MPI_Isend(&buffE[0], ny, MPI_DOUBLE, domain.neighbour_east, E, comm_cart, &sndE_req);
    }

    if (domain.neighbour_west >= 0)
    {
        for (int j = 0; j <= jend; j++)
            buffW[j] = U(iend, j);
        MPI_Irecv(&bndW[0], ny, MPI_DOUBLE, domain.neighbour_west, tag_from(W), comm_cart, &rcvW_req);
        MPI_Isend(&buffW[0], ny, MPI_DOUBLE, domain.neighbour_west, W, comm_cart, &sndW_req);
    }
    MPI_Status status;
    if (domain.neighbour_north >= 0){
        MPI_Wait(&sndN_req, &status);
        MPI_Wait(&rcvN_req, &status);
    }
    if (domain.neighbour_south >= 0){
        MPI_Wait(&sndS_req, &status);
        MPI_Wait(&rcvS_req, &status);
    }
    if (domain.neighbour_east >= 0){
        MPI_Wait(&sndE_req, &status); 
        MPI_Wait(&rcvE_req, &status);

    }
    if (domain.neighbour_west >= 0){
        MPI_Wait(&sndW_req, &status);
        MPI_Wait(&rcvW_req, &status);
    }

    // the interior grid points
    #pragma omp parallel for
    for (int j=1; j < jend; j++) {
        for (int i=1; i < iend; i++) {
            S(i,j) = -(4. + alpha) * U(i,j)               // central point
                                    + U(i-1,j) + U(i+1,j) // east and west
                                    + U(i,j-1) + U(i,j+1) // north and south
                                    + alpha * x_old(i,j)
                                    + dxs * U(i,j) * (1.0 - U(i,j));
        }
    }

    // the east boundary
    {
        int i = nx - 1;
        for (int j = 1; j < jend; j++)
        {
            S(i,j) = -(4. + alpha) * U(i,j)
                        + U(i-1,j) + U(i,j-1) + U(i,j+1)
                        + alpha*x_old(i,j) + bndE[j]
                        + dxs * U(i,j) * (1.0 - U(i,j));
        }
    }

    // the west boundary
    {
        int i = 0;
        for (int j = 1; j < jend; j++)
        {
            S(i,j) = -(4. + alpha) * U(i,j)
                        + U(i+1,j) + U(i,j-1) + U(i,j+1)
                        + alpha * x_old(i,j) + bndW[j]
                        + dxs * U(i,j) * (1.0 - U(i,j));
        }
    }

    // the north boundary (plus NE and NW corners)
    {
        int j = ny - 1;

        {
            int i = 0; // NW corner
            S(i,j) = -(4. + alpha) * U(i,j)
                        + U(i+1,j) + U(i,j-1)
                        + alpha * x_old(i,j) + bndW[j] + bndN[i]
                        + dxs * U(i,j) * (1.0 - U(i,j));
        }

        // north boundary
        for (int i = 1; i < iend; i++)
        {
            S(i,j) = -(4. + alpha) * U(i,j)
                        + U(i-1,j) + U(i+1,j) + U(i,j-1)
                        + alpha*x_old(i,j) + bndN[i]
                        + dxs * U(i,j) * (1.0 - U(i,j));
        }

        {
            int i = nx-1; // NE corner
            S(i,j) = -(4. + alpha) * U(i,j)
                        + U(i-1,j) + U(i,j-1)
                        + alpha * x_old(i,j) + bndE[j] + bndN[i]
                        + dxs * U(i,j) * (1.0 - U(i,j));
        }
    }

    // the south boundary
    {
        int j = 0;

        {
            int i = 0; // SW corner
            S(i,j) = -(4. + alpha) * U(i,j)
                        + U(i+1,j) + U(i,j+1)
                        + alpha * x_old(i,j) + bndW[j] + bndS[i]
                        + dxs * U(i,j) * (1.0 - U(i,j));
        }

        // south boundary
        for (int i = 1; i < iend; i++)
        {
            S(i,j) = -(4. + alpha) * U(i,j)
                        + U(i-1,j) + U(i+1,j) + U(i,j+1)
                        + alpha * x_old(i,j) + bndS[i]
                        + dxs * U(i,j) * (1.0 - U(i,j));
        }

        {
            int i = nx - 1; // SE corner
            S(i,j) = -(4. + alpha) * U(i,j)
                        + U(i-1,j) + U(i,j+1)
                        + alpha * x_old(i,j) + bndE[j] + bndS[i]
                        + dxs * U(i,j) * (1.0 - U(i,j));
        }
    }

    // Accumulate the flop counts
    // 8 ops total per point
    stats::flops_diff +=
        + 12 * (nx - 2) * (ny - 2) // interior points
        + 11 * (nx - 2  +  ny - 2) // NESW boundary points
        + 11 * 4;                                  // corner points
}

} // namespace operators
