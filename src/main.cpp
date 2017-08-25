#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <ctime>
#include <cmath>
#include <complex>

#define EIGEN_USE_MKL_ALL

#include <omp.h>
#include <Eigen/PardisoSupport>


typedef Eigen::SparseMatrix<double> SpMat; // declares a column-major sparse matrix type of double
typedef Eigen::Triplet<double> T;
typedef std::numeric_limits< long double > dbl;

#include "getRAMusage.hpp"
#include "utils.hpp"
#include "RealValuedSolverLLT.hpp"
#include "RealValuedSolverLDLT.hpp"
#include "RealValuedSolverPardiso.hpp"

// Eigenvalues of sparse matrix
#include <GenEigsSolver.h>
#include <MatOp/SparseGenMatProd.h>
#include <SymEigsSolver.h>  // Also includes <MatOp/DenseSymMatProd.h>
#include <SymEigsShiftSolver.h>  // Also includes <MatOp/DenseSymShiftSolve.h>
#include <MatOp/SparseSymShiftSolve.h>




int main(int argc, char** argv)
{
  // assert(argc >= 2);
  // omp_set_num_threads(4);
  // Eigen::setNbThreads(4);
  std::cout << "OpenMP settings : " << Eigen::nbThreads() << " threads running "<< '\n';

  // int nx = 3;
  // if (argc >= 2) {
  //   nx = atoi(argv[1]);
  // }

  // int n = pow(nx,2);
  // std::cout << "n  " << n << std::endl;
  //
  // Eigen::VectorXd phi(n);  Eigen::VectorXd psi(n);
  // Eigen::VectorXd x(n);  Eigen::VectorXd y(n);
  // SpMat R(n,n);  SpMat S(n,n);
  // std::vector<T> coefficients_R, coefficients_S;
  //
  // buildRVProblem(coefficients_R, coefficients_S, phi,  psi, nx);
  // std::cout << "normes entrées " << phi.norm() << " " << psi.norm() << '\n';
  /////////////////////////////////////////////////////////////////////////////
  // int n = 3;
  // int n = 27623;
  // int n = 181302;


  // M*e'(t) = C*e(t),  M*e(0) = q,
  std::string file_name1 = "./ressources/TEM27623/TEM27623_C.mtx"; // TEM Symmetrix positive
  std::string file_name2 = "./ressources/TEM27623/TEM27623_M.mtx"; // TEM Symmetrix positive definite (invertible)

  // std::string file_name1 = "./ressources/TEM27623/TEM27623_C.mtx"; // Symmetrix positive
  // std::string file_name2 = "./ressources/TEM27623/TEM27623_M.mtx"; // Symmetrix positive definite (invertible)

  if (argc == 2) {
    file_name1 = std::string(argv[1]);
    file_name2 = std::string(argv[1]);
  }
  std::cout << '\n';

  std::complex<double> xi = std::complex<double>(-1, -1);

  SpMat K, L;
  read_matrix_MtxMarket(file_name1, L);
  read_matrix_MtxMarket(file_name2, K);

  int n = L.rows();
  std::cout << "Working with " << n << "x" << n << " matrices" << '\n';
  Eigen::VectorXd phi =  Eigen::MatrixXd::Random(n, 1); phi = phi / phi.norm(); //phi.setZero();
  Eigen::VectorXd psi =  Eigen::MatrixXd::Random(n, 1); psi = psi / psi.norm(); //psi.setZero();
  std::cout << "Norme L : " << L.norm() << '\n';
  std::cout << "Norme K : " << K.norm() << '\n';


  Eigen::SparseMatrix<std::complex<double > > A(n,n);
  Eigen::VectorXcd u(n);
  Eigen::VectorXcd b(n);
  A = xi * K.cast<std::complex<double > >() - L.cast<std::complex<double > >();
  b.real() = phi;
  b.imag() = psi;
  // b = phi;
  std::cout << "Norme de A : " << A.norm() << std::endl << std::endl;


  //////////////////////////////////////////////////////////////////////////////
  ///                Testing matrices                                        ///
  //////////////////////////////////////////////////////////////////////////////

  std::cout << "=== Matrix K" << '\n';
  Spectra::SparseGenMatProd<double> op_k(K);
  Spectra::SymEigsSolver< double, Spectra::LARGEST_MAGN, Spectra::SparseGenMatProd<double> > eigs_k_max(&op_k, 3, 6);
  eigs_k_max.init();
  int nconv = eigs_k_max.compute();
  Eigen::VectorXcd evalues;
  if(eigs_k_max.info() == Spectra::SUCCESSFUL)
      evalues = eigs_k_max.eigenvalues();
  std::cout << "Largest eigenvalues found:\t" << evalues[0] << "\t" << evalues[1] << "\t" << evalues[2] << "\t" << std::endl;

  Spectra::SparseSymShiftSolve<double> op_k_min(K);
  Spectra::SymEigsShiftSolver< double, Spectra::LARGEST_MAGN, Spectra::SparseSymShiftSolve<double> > eigs_k_min(&op_k_min, 3, 6, 0.0);
  eigs_k_min.init();
  nconv = eigs_k_min.compute();
  if(eigs_k_min.info() == Spectra::SUCCESSFUL)
      evalues = eigs_k_min.eigenvalues();
  std::cout << "Smallest eigenvalues found:\t" << evalues[0] << "\t" << evalues[1] << "\t" << evalues[2] << "\t" << std::endl;

  Eigen::SparseMatrix<double> Ktr = K.transpose();
  std::cout << "(K-K.transpose()).norm() : " << (K-Ktr).norm() << '\n' << std::endl;

  std::cout << "=== Matrix L" << '\n';
  Spectra::SparseGenMatProd<double> op_l(L);
  Spectra::SymEigsSolver< double, Spectra::LARGEST_MAGN, Spectra::SparseGenMatProd<double> > eigs_l_max(&op_l, 3, 6);
  eigs_l_max.init();
  nconv = eigs_l_max.compute();
  if(eigs_l_max.info() == Spectra::SUCCESSFUL)
      evalues = eigs_l_max.eigenvalues();
  std::cout << "Largest eigenvalues found:\t" << evalues[0] << "\t" << evalues[1] << "\t" << evalues[2] << "\t" << std::endl;

  Spectra::SparseSymShiftSolve<double> op_l_min(K);
  Spectra::SymEigsShiftSolver< double, Spectra::LARGEST_MAGN, Spectra::SparseSymShiftSolve<double> > eigs_l_min(&op_l_min, 3, 6, 0.0);
  eigs_l_min.init();
  nconv = eigs_l_min.compute();
  if(eigs_l_min.info() == Spectra::SUCCESSFUL)
      evalues = eigs_l_min.eigenvalues();
  std::cout << "Smallest eigenvalues found:\t" << evalues[0] << "\t" << evalues[1] << "\t" << evalues[2] << "\t" << std::endl;

  Eigen::SparseMatrix<double> Ltr = L.transpose();
  std::cout << "(L-L.transpose()).norm() : " << (L-Ltr).norm() << '\n' << std::endl;

  std::cout << "=== Matrix K" << '\n';
  info_matrix(K);

  std::cout << "=== Matrix L" << '\n';
  info_matrix(L);

  //////////////////////////////////////////////////////////////////////////////
  ///                SolverRealValued Object test                            ///
  //////////////////////////////////////////////////////////////////////////////

  std::cout << "===== In main declaration...\n" << '\n';

  Eigen::SimplicialLLT<Eigen::SparseMatrix<double> > m_solverBAlpha;
  Eigen::SparseMatrix<double> My_matrix = L + 2*K;

  std::cout << "=== Matrix L+2*K" << '\n';
  info_matrix(My_matrix);

  m_solverBAlpha.compute(My_matrix);

  if (m_solverBAlpha.info() == Eigen::InvalidInput) {
    std::cout << "Factorize : Invalid Input" << '\n';
  } else if (m_solverBAlpha.info() == Eigen::NoConvergence) {
    std::cout << "Factorize : No Convergence" << '\n';
  } else if (m_solverBAlpha.info() == Eigen::NumericalIssue) {
    std::cout << "Factorize : NumericalIssue" << '\n';
  } else {
    std::cout << "factorize : ok" << '\n';
  }


  std::cout << "============================= RVAlgo Eigen solve...\n" << '\n';


  // double time_ref = omp_get_wtime();
  // double tol = 1e-16;
  clock_t start_opt = clock();
  RealValuedSolverLLT solverRV(-1*A.real(), -1*A.imag());
  u = solverRV.solve(-K*b);

  solverRV.info();
  std::cout << std::endl << "RVsolver (full cpu_time)  :  " << ( std::clock() - start_opt ) / (double) CLOCKS_PER_SEC << " s" << std::endl;
  // std::cout << "RVsolver (obs proper time):  " << omp_get_wtime() - time_ref << " s" << std::endl;
  std::cout << "Overall residual :  " << sqrt((A*u-K*b).real().squaredNorm() + (A*u-K*b).imag().squaredNorm()) << std::endl << std::endl;
  std::cout << "u norm " << u.real().norm() << " " << u.imag().norm() << '\n';
  u.setZero();

  // std::cout << "============================= RVAlgo Eigen solve for multiple RHS..." << '\n';
  //
  // start_opt = clock();
  // // time_ref = omp_get_wtime();
  // RealValuedSolverLLT solverRVM(R, S, tol);
  // Eigen::MatrixXcd U = solverRVM.solve(B);
  //
  // solverRVM.info();
  // std::cout << std::endl << "RVsolver (full cpu_time)  :  " << ( std::clock() - start_opt ) / (double) CLOCKS_PER_SEC << " s" << std::endl;
  // // std::cout << "RVsolver (obs proper time):  " << omp_get_wtime() - time_ref << " s" << std::endl;
  // std::cout << "Overall residual (1) :  " << sqrt((A*U.col(0)-B.col(0)).real().squaredNorm() + (A*U.col(0)-B.col(0)).imag().squaredNorm()) << std::endl;
  // std::cout << "Overall residual (2) :  " << sqrt((A*U.col(1)-B.col(1)).real().squaredNorm() + (A*U.col(1)-B.col(1)).imag().squaredNorm()) << std::endl << std::endl;
  // std::cout << "U norm " << U.norm() << '\n';
  // u.setZero();
  //
  // //////////////////////////////////////////////////////////////////////////////
  // ///                SolverRealValued Pardiso test                           ///
  // //////////////////////////////////////////////////////////////////////////////
  //
  // std::cout << "============================= RVAlgo Pardiso solve..." << '\n';
  //
  // start_opt = clock();
  // // time_ref = omp_get_wtime();
  // RealValuedSolverPardiso solverRVPds(R, S, tol);
  // u = solverRVPds.solve(b);
  //
  // solverRVPds.info();
  // std::cout << std::endl << "RVsolver Pardiso (full cpu_time)  :  " << ( std::clock() - start_opt ) / (double) CLOCKS_PER_SEC << " s" << std::endl;
  // // std::cout << "RVsolver Pardiso (obs proper time):  " << omp_get_wtime() - time_ref << " s" << std::endl;
  // std::cout << "Overall residual :  " << sqrt((A*u-b).real().squaredNorm() + (A*u-b).imag().squaredNorm()) << std::endl << std::endl;
  // std::cout << "u norm " << u.norm() << '\n';
  // u.setZero();

  //////////////////////////////////////////////////////////////////////////////
  ///                     Pardiso direct solver                              ///
  //////////////////////////////////////////////////////////////////////////////

  std::cout << "============================= Direct Pardiso LDLT solve..." << '\n';
  // time_ref = omp_get_wtime();
  clock_t overall_clock = clock();
  start_opt = clock();
  Eigen::PardisoLDLT<Eigen::SparseMatrix<std::complex<double>>, Eigen::Symmetric | Eigen::Upper> solverPardiso;
  std::cout << "definition  : " << ( std::clock() - start_opt ) / (double) CLOCKS_PER_SEC << " s" << std::endl;
  start_opt = clock();
  solverPardiso.compute(A);
  std::cout << "compute     : " << ( std::clock() - start_opt ) / (double) CLOCKS_PER_SEC << " s" << std::endl;
  start_opt = clock();
  Eigen::VectorXcd kb;
  kb = K*b;
  Eigen::VectorXcd u_pardiso = solverPardiso.solve(kb);
  std::cout << "solve       : " << ( std::clock() - start_opt ) / (double) CLOCKS_PER_SEC << " s" << std::endl;
  start_opt = clock();

  std::cout << std::endl << "Pardiso (full cpu_time)  :  " << ( std::clock() - overall_clock ) / (double) CLOCKS_PER_SEC << " s" << std::endl;
  // std::cout << "Pardiso (obs proper time):  " << omp_get_wtime() - time_ref << " s" << std::endl;
  std::cout << "Overall residual :  " << sqrt((A*u_pardiso-K*b).real().squaredNorm() + (A*u_pardiso-b).imag().squaredNorm()) << std::endl << std::endl;
  std::cout << "(u - u_pardiso) norm " << (u-u_pardiso).norm() << '\n';

  printf("my print : %10.64f\n", K.coeffRef(0,0));
  return 0;
}
