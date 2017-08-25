#include <mex.h>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <Eigen/SparseCholesky>

#include "utilsMex.hpp"
#include "RealValuedSolver.hpp"

// mex -v GCC='/usr/bin/g++-4.7' CXXFLAGS="\$CXXFLAGS -std=c++11" -I"./include/" -largeArrayDims RVsolver.cpp include/utilsMex.cpp include/RealValuedSolver.cpp

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // Expected input :
        // - A : sparse double matrix
        // - K : sparse double matrix
        // - b : right hand side vector
        // - xis : double complex vector
    // Solves (xi*K-A)u = b for every xi, when K, A are semi-positive definite and posi-
    // tive definite respectively

    /* Check for proper number of input and output arguments */
    if (nrhs != 4) {
        mexErrMsgIdAndTxt( "MATLAB:readSparse:invalidNumInputs",
                "Four input argument required.\nCall [linSols] = blackbox(A, K, b, xis)");
    }
    if(nlhs > 2){
        mexErrMsgIdAndTxt( "MATLAB:readSparse:maxlhs",
                "Too many output arguments.");
    }

    // double tol = 1e-16;

    // Reading input
    Eigen::SparseMatrix<double> K, A;
    Eigen::MatrixXd b;
    std::vector<std::complex<double>> xis;

    std::cout << "Print b" << '\n';
    for (int i=0;i<b.rows();++i) {
      for (int j=0;j<b.cols();++j) {
        std::cout << b(i,j) << " ";
      }
      std::cout << "" << '\n';
    }

    readSparse(prhs[0], A);
    readSparse(prhs[1], K);
    readRealMatrix(prhs[2], b);
    readComplexArray(prhs[3], xis);


    std::vector<Eigen::MatrixXcd> solsLinSyst(xis.size());
    for (int i=0;i<xis.size();++i) solsLinSyst[0].resize(b.rows(), b.cols());
    // solsLinSyst.resize(b.size(), xis.size());

    std::complex<double> xi;
    for (int i=0;i<xis.size();++i) {
      xi = xis[i];
      std::cout << "It " << i << ", xi=" << xi << ",\t";
      RealValuedSolver solver(A-xi.real()*K, -xi.imag()*K);
      solver.compute();
      std::cout << "compute done" << '\n';
      Eigen::MatrixXcd sol = solver.solve(b);
      // std::cout << "xi is real ! Solving directly" << '\n';
      // Eigen::SimplicialLDLT<SpMat > solverRealSyst;
      // solverRealSyst.compute(A-xi.real()*K);
      // sol = solverRealSyst.solve(b);
      solver.info();
      std::cout << "residual: " << solver.timings()[6] << '\n';
      if (solver.timings()[6] > 1e-10) mexWarnMsgTxt("Solution residual too high.");
      solsLinSyst[i] = sol;
    }

    // std::cout << "Saving solution now, " << solsLinSyst.rows() << "x" << solsLinSyst.cols() << '\n';
    // plhs[0] = writeComplexMatrix(solsLinSyst);


    std::vector<std::complex<double> > v;
    // int dims[3] = {2,2,2};
    std::vector<size_t> dims;
    dims.push_back(b.rows());
    dims.push_back(2); // b.cols()
    dims.push_back(xis.size());

    std::cout << "dims : " << dims[0] << " " << dims[1] << " " << dims[2] << " " << '\n';

    plhs[0] = mxCreateNumericArray(3, dims.data(), mxDOUBLE_CLASS, mxREAL);
    double* pr = mxGetPr(plhs[0]);

    for (int ind_xi=0;ind_xi<dims[2];++ind_xi) {
      for (int ind_col=0;ind_col<dims[1];++ind_col) { // b.cols()
        for (int ind_row=0;ind_row<dims[0];++ind_row) {
          pr[ind_xi*dims[1]*dims[0] + ind_col*dims[0] + ind_row] = ind_xi*dims[1]*dims[0] + ind_col*dims[0] + ind_row;
        }
      }
    }

    return;
}
