#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <complex>

#include <Eigen/Sparse>
#include <Eigen/Dense>


typedef Eigen::SparseMatrix<double> SpMat; // declares a column-major sparse matrix type of double
typedef Eigen::Triplet<double> T;
typedef std::numeric_limits< long double > dbl;

#include "RealValuedSolver.hpp"
#include "utils.hpp"

int main(int argc, char** argv)
{
  assert(argc >= 2);

  int nx = 3;
  if (argc >= 2) {
    nx = atoi(argv[1]);
  }
  int n = pow(nx,2);
  std::cout << "n  " << n << std::endl;

  Eigen::VectorXd phi(n);  Eigen::VectorXd psi(n);
  // Eigen::VectorXd x(n);  Eigen::VectorXd y(n);
  // SpMat R(n,n);  SpMat S(n,n);

  std::vector<T> coefficients_R, coefficients_S;
  buildRVProblem(coefficients_R, coefficients_S, phi,  psi, nx);
  std::cout << "normes entrées " << phi.norm() << " " << psi.norm() << '\n';


  // Eigen::SparseMatrix<std::complex<double > > A(n,n);
  // Eigen::VectorXcd u(n);

  // std::cout << "Building complex matrix and vector" << '\n';
  // A = R.cast<std::complex<double > >() + std::complex<double>(0.0,1.0) * S.cast<std::complex<double > >();
  Eigen::VectorXd b;
  // b = phi.cast<std::complex<double > >() + std::complex<double>(0.0,1.0) * psi.cast<std::complex<double > >();
  // std::cout << "Norme de A : " << A.norm() << " norme de b " << b.norm() << '\n';

  //////////////////////////////////////////////////////////////////////////////
  ///                SolverRealValued Object test                            ///
  //////////////////////////////////////////////////////////////////////////////

  // Reading input
  Eigen::SparseMatrix<double> K(n,n);
  Eigen::SparseMatrix<double> A(n,n);
  K.setFromTriplets(coefficients_R.begin(), coefficients_R.end());
  A.setFromTriplets(coefficients_S.begin(), coefficients_S.end());

  b = phi;
  std::cout << "norme de b " << b.norm() << '\n';

  Eigen::MatrixXd rhs, x;
  rhs.resize(b.size(),2);
  rhs.col(0) = b;
  rhs.col(1) = 3*b;

  // std::cout << "------------------------------------------------------" << '\n';
  // // fill A and b
  // Eigen::ConjugateGradient<Eigen::SparseMatrix<double>, Eigen::Lower|Eigen::Upper> cg;
  // cg.compute(A);
  // std::cout << "rhs : " << rhs << '\n';
  // x = cg.solve(rhs);
  // std::cout << "#iterations:     " << cg.iterations() << std::endl;
  // std::cout << "estimated error: " << cg.error()      << std::endl;
  // // update b, and solve again
  // x = cg.solve(b);
  //
  // std::cout << "res1 : " << (A*x-rhs.col(0)).norm() << " " << (A*x-rhs.col(1)).norm() << '\n';
  // std::cout << "solution x " << '\n' << x << '\n';
  //
  // std::cout << "------------------------------------------------------" << '\n';

  std::vector<std::complex<double>> xis;

  // readSparse(prhs[0], A);
  // readSparse(prhs[1], K);
  // readRealVector(prhs[2], b);
  // readComplexArray(prhs[3], xis);

  // xis.push_back(std::complex<double>(-0.8151,1.4669));
  // xis.push_back(std::complex<double>(-0.8151,-1.4669));
  // xis.push_back(std::complex<double>(-1.0836,4.3985));
  xis.push_back(std::complex<double>(-3.0836,4.3985));
  // xis.push_back(std::complex<double>(-1.0836,-4.3985));

  Eigen::MatrixXcd solsLinSyst;
  solsLinSyst.resize(b.size(), xis.size());

  // Eigen::MatrixXd rhs;
  rhs.resize(b.size(),2);
  rhs.col(0) = b;
  rhs.col(1) = 3*b;
  Eigen::MatrixXd sol_real(rhs.rows(), rhs.cols());
  Eigen::MatrixXd sol_imag(rhs.rows(), rhs.cols());

  std::complex<double> xi;
  for (unsigned int i=0;i<xis.size();++i) {
    xi = xis[i];
    std::cout << "--------------------------------------------------------------" << '\n';
    std::cout << "----------- Iteration " << i << ", xi=" << xi << '\n';
    RealValuedSolver solver(xi.real()*K-A, xi.imag()*K);
    solver.compute();
    solver.solve(rhs, sol_real, sol_imag);
    std::cout << "residual : " << solver.timings()[6] << '\n';
    solsLinSyst.col(i) = sol_real;
  }

  std::cout << solsLinSyst << '\n';

  return 0;
}
