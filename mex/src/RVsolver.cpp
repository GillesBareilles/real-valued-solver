#include <mex.h>
#include <Eigen/Sparse>
#include <Eigen/Dense>

#include "utilsMex.hpp"
#include "RealValuedSolver.hpp"

// mex -v GCC='/usr/bin/g++-4.7' CXXFLAGS="\$CXXFLAGS -std=c++11" -I"./include/" -largeArrayDims RVsolver.cpp include/utilsMex.cpp include/RealValuedSolver.cpp

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // Expected input :
        // - R : sparse double matrix
        // - S : sparse double matrix
        // - b : double vector
    // Solves (R+iS)u = b for u, when R, S are positive definite and semi posi-
    // tive definite respectively

    /* Check for proper number of input and output arguments */
    if (nrhs > 4) {
        mexErrMsgIdAndTxt( "MATLAB:readSparse:invalidNumInputs",
                "Three input argument required.");
    }
    if(nlhs > 3){
        mexErrMsgIdAndTxt( "MATLAB:readSparse:maxlhs",
                "Too many output arguments.");
    }

    // TODO : test type of input

    // TODO : test R and S for DP and SDP, raise error in case not.
    // NOTE : take a look at Krylov methods to estimate smallest and largest
    // eigenvalues.

    double tol = 1e-16;

    if (nrhs = 4) {
      std::vector<double> temp;
      readRealArray(prhs[2], temp);
      tol = temp[0];
    }

    Eigen::SparseMatrix<double> R, S;
    Eigen::VectorXd b;

    readSparse(prhs[0], R);
    readSparse(prhs[1], S);
    readRealVector(prhs[2], b);

    RealValuedSolver solver(R, S, tol);
    Eigen::VectorXcd sol = solver.solve(b);

    plhs[0] = writeComplexVector(sol);

    // std::vector<double> error;
    // error.push_back(sqrt((R*sol.real() - S*sol.imag() - b).squaredNorm() + (R*sol.imag()+S*sol.real()).squaredNorm()));
    // plhs[1] = writeRealArray(error);
    plhs[1] = writeRealArray(solver.timings());
    plhs[2] = writeRealArray(solver.CG_info());

    // mexPrintf("Absolute error : %g\t", sqrt((R*sol.real() - S*sol.imag() - b).squaredNorm() + (R*sol.imag()+S*sol.real()).squaredNorm()));
    // mexPrintf("Solution norm : %g\n", sol.norm());

    return;
}
