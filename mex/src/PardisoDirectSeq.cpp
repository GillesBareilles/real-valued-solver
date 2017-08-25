#include <mex.h>

#define EIGEN_USE_MKL_ALL
#include <omp.h>
#include <Eigen/PardisoSupport>

// #include <Eigen/Sparse>
// #include <Eigen/Dense>

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
    if (nrhs != 3) {
        mexErrMsgIdAndTxt( "MATLAB:readSparse:invalidNumInputs",
                "Three input argument required.\nCall [sol] = PardisoDirectSeq(R, S, b)");
    }
    if(nlhs > 2){
      mexErrMsgIdAndTxt( "MATLAB:readSparse:maxlhs",
                "Too many output arguments.");
    }

    Eigen::SparseMatrix<double> R, S;
    Eigen::VectorXcd b;

    readSparse(prhs[0], R);
    readSparse(prhs[1], S);
    readComplexVector(prhs[2], b);

    Eigen::VectorXcd(b.rows());
    std::vector<double> info;

    std::cout << "Direct Pardiso LDLT solve :" << '\n';
    // double time_ref = omp_get_wtime();
    // clock_t overall_clock = clock();
    // clock_t start_opt = clock();
    // Eigen::PardisoLDLT<Eigen::SparseMatrix<std::complex<double>>, Eigen::Symmetric | Eigen::Upper> solverPardiso;
    // info.push_back(( std::clock() - start_opt ) / (double) CLOCKS_PER_SEC);
    // std::cout << "definition  : " << ( std::clock() - start_opt ) / (double) CLOCKS_PER_SEC << " s" << std::endl;
    //
    // start_opt = clock();
    // Eigen::SparseMatrix<std::complex<double>> A(R.rows(), R.cols());
    // A = R.cast<std::complex<double > >() + std::complex<double>(0.0,1.0) * S.cast<std::complex<double > >();
    // solverPardiso.compute(A);
    // info.push_back(( std::clock() - start_opt ) / (double) CLOCKS_PER_SEC);
    // std::cout << "compute     : " << ( std::clock() - start_opt ) / (double) CLOCKS_PER_SEC << " s" << std::endl;
    //
    // start_opt = clock();
    // Eigen::VectorXcd sol = solverPardiso.solve(b);
    // info.push_back(( std::clock() - start_opt ) / (double) CLOCKS_PER_SEC);
    // std::cout << "solve       : " << ( std::clock() - start_opt ) / (double) CLOCKS_PER_SEC << " s" << std::endl;
    //
    // info.push_back(( std::clock() - overall_clock ) / (double) CLOCKS_PER_SEC);
    // std::cout << std::endl << "Pardiso (full cpu_time)  :  " << ( std::clock() - overall_clock ) / (double) CLOCKS_PER_SEC << " s" << std::endl;
    // // std::cout << "Pardiso (obs proper time):  " << omp_get_wtime() - time_ref << " s" << std::endl;
    // info.push_back(sqrt((R*sol.real() - S*sol.imag()-b).squaredNorm() + (R*sol.imag()+S*sol.real()).squaredNorm()));
    // std::cout << "Overall residual :  " << info[4] << std::endl << std::endl;
    //
    // plhs[0] = writeComplexVector(sol);
    // plhs[1] = writeRealArray(info);

    return;
}
