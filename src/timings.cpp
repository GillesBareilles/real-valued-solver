#include <iostream>
#include <fstream>

#include <vector>
#include <string>
#include <ctime>
#include <cmath>
#include <complex>
#include <ctime> // day and time

#define EIGEN_USE_MKL_ALL

#include <omp.h>
#include <Eigen/PardisoSupport>


typedef Eigen::SparseMatrix<double> SpMat; // declares a column-major sparse matrix type of double
typedef Eigen::Triplet<double> T;
typedef std::numeric_limits< long double > dbl;
typedef Eigen::Matrix<double, 12, 1> MatrixInfo;

#include "getRAMusage.hpp"
#include "utils.hpp"
#include "RealValuedSolverLLT.hpp"
#include "RealValuedSolverLDLT.hpp"
#include "RealValuedSolverPardiso.hpp"

void runAlgorithms(const Eigen::SparseMatrix<double>& R, const Eigen::SparseMatrix<double>& S,
  const Eigen::VectorXd& phi, const Eigen::VectorXd& psi, MatrixInfo& v_solverRV_LDLT,
  MatrixInfo& v_solverRV_LLT, MatrixInfo& v_solverRV_Pardiso,
  MatrixInfo& v_directPardiso);

int main(int argc, char** argv)
{
  // assert(argc >= 2);
  omp_set_num_threads(4);
  Eigen::setNbThreads(4);

  // get time of start
  time_t tStart = time(NULL);
	tm* timeStartPtr = localtime(&tStart);
  std::string dateStart =  std::to_string(timeStartPtr->tm_mon) + "." + std::to_string(timeStartPtr->tm_mday) + "-" + std::to_string(timeStartPtr->tm_hour) + ":" + std::to_string(timeStartPtr->tm_min) + ":" + std::to_string(timeStartPtr->tm_sec);

  std::cout << "OpenMP settings : " << Eigen::nbThreads() << " threads running "<< '\n';

  int nx_min = 10;
  int nx_max = 200;
  int nb_nx = 10;
  if (argc>2) nx_min = atoi(argv[2]);
  if (argc>3) nx_max = atoi(argv[3]);
  if (argc>4) nb_nx = atoi(argv[4]);

  // Eigen::VectorXd nx_range = Eigen::VectorXd::LinSpaced(nb_nx,log(nx_min),log(nx_max));
  Eigen::VectorXd nx_range = Eigen::VectorXd::LinSpaced(3,log(10),log(10));
  nx_range = nx_range.array().exp().ceil();
  // nx_range = Eigen::VectorXd::LinSpaced(7,10,10);

  std::cout << "Timing for nx spanning : ";
  for (int i=0;i<nx_range.size();i++) {
    std::cout << nx_range(i) << ' ';
  }
  std::cout << '\n';

  // Data structures
  Eigen::MatrixXd data_solverRV_LLT(12, nx_range.size()); data_solverRV_LLT.setZero();
  Eigen::MatrixXd data_solverRV_LDLT(12, nx_range.size()); data_solverRV_LDLT.setZero();
  Eigen::MatrixXd data_solverRV_Pardiso(12, nx_range.size()); data_solverRV_Pardiso.setZero();
  Eigen::MatrixXd data_directPardiso(12, nx_range.size()); data_directPardiso.setZero();

  MatrixInfo v_solverRV_LLT;
  MatrixInfo v_solverRV_LDLT;
  MatrixInfo v_solverRV_Pardiso;
  MatrixInfo v_directPardiso;

  int nbRepetitions = 1;
  if (argc>1) {
    nbRepetitions = atoi(argv[1]);
    std::cout << "repeating " << nbRepetitions << " times"<< '\n';
  }

  for (int rep=0;rep<nbRepetitions;rep++) {
    v_solverRV_LLT.setZero();
    v_solverRV_LDLT.setZero();
    v_solverRV_Pardiso.setZero();
    v_directPardiso.setZero();

    //////////////////////////////////////////////////////////////////////////
    // Geophysics
    //////////////////////////////////////////////////////////////////////////
    for(int j=0;j<nx_range.size();++j) {
      std::cout << "Rep " << rep+1 << "/" << nbRepetitions << ", j=" << j+1 << "/" << nx_range.size() << '\n';

      int n=0;
      if (j == 0) n = 27623;
      else if (j == 1) n = 152078;
      else if (j == 2) n = 181302;

      // Defining matrices and vectors
      Eigen::VectorXd phi(n);  Eigen::VectorXd psi(n);
      phi.setZero(); psi.setZero();
      SpMat R(n,n);  SpMat S(n,n);
      std::vector<T> coefficients_R, coefficients_S;
      read_tem_matrices(coefficients_R, coefficients_S, phi, n);
      std::cout << "Norme phi : " << phi.norm() << '\n';

      // std::cout << std::endl << "normes entrées " << phi.norm() << " " << psi.norm() << '\n';
      // std::cout << "Building matrices..." << '\n';
      R.setFromTriplets(coefficients_R.begin(), coefficients_R.end());
      S.setFromTriplets(coefficients_S.begin(), coefficients_S.end());
      std::cout << "Norme R : " << R.norm() << '\n';
      std::cout << "Norme S : " << S.norm() << '\n';

      std::cout << "R matrix : " << '\n' << R.block(0,0,5,5);
      // R and S should be SPD
      // Eigen::SparseMatrix<double> Rtr = R.transpose();
      // std::cout << "(R-R.transpose()).norm() : " << (R-Rtr).norm() << '\n';
      // Eigen::SparseMatrix<double> Str = S.transpose();
      // std::cout << "(S-S.transpose()).norm() : " << (S-Str).norm() << '\n';

      // Running algorithms
      runAlgorithms(R+S, R+S, phi, psi, v_solverRV_LLT, v_solverRV_LDLT,
                    v_solverRV_Pardiso, v_directPardiso);

      // Storing results

      // std::cout << "data_solverRV_LLT \t" << v_solverRV_LLT << '\n';
      // std::cout << "data_solverRV_LDLT \t" << v_solverRV_LDLT << '\n';
      // std::cout << "data_solverRV_Pardiso \t" << v_solverRV_Pardiso << '\n';
      // std::cout << "data_directPardiso \t" << v_directPardiso << '\n';

      data_solverRV_LLT.col(j) += v_solverRV_LLT/nbRepetitions;
      data_solverRV_LDLT.col(j) += v_solverRV_LDLT/nbRepetitions;
      data_solverRV_Pardiso.col(j) += v_solverRV_Pardiso/nbRepetitions;
      data_directPardiso.col(j) += v_directPardiso/nbRepetitions;
    }

    //////////////////////////////////////////////////////////////////////////
    // for(int j=0;j<nx_range.size();++j) {
    //   std::cout << "Rep " << rep+1 << "/" << nbRepetitions << ", j=" << j+1 << "/" << nx_range.size() << '\n';
    //
    //   int nx=nx_range(j);
    //   int n=pow(nx,2);
    //
    //   // Defining matrices and vectors
    //   Eigen::VectorXd phi(n);  Eigen::VectorXd psi(n);
    //   SpMat R(n,n);  SpMat S(n,n);
    //   std::vector<T> coefficients_R, coefficients_S;
    //   buildRVProblem(coefficients_R, coefficients_S, phi,  psi, nx);
    //
    //   // std::cout << std::endl << "normes entrées " << phi.norm() << " " << psi.norm() << '\n';
    //   // std::cout << "Building matrices..." << '\n';
    //   R.setFromTriplets(coefficients_R.begin(), coefficients_R.end());
    //   S.setFromTriplets(coefficients_S.begin(), coefficients_S.end());
    //
    //   // R and S should be SPD
    //   // Eigen::SparseMatrix<double> Rtr = R.transpose();
    //   // std::cout << "(R-R.transpose()).norm() : " << (R-Rtr).norm() << '\n';
    //   // Eigen::SparseMatrix<double> Str = S.transpose();
    //   // std::cout << "(S-S.transpose()).norm() : " << (S-Str).norm() << '\n';
    //
    //   // Running algorithms
    //   runAlgorithms(R, S, phi, psi, v_solverRV_LLT, v_solverRV_LDLT,
    //                 v_solverRV_Pardiso, v_directPardiso);
    //
    //   // Storing results
    //
    //   // std::cout << "data_solverRV_LLT \t" << v_solverRV_LLT << '\n';
    //   // std::cout << "data_solverRV_LDLT \t" << v_solverRV_LDLT << '\n';
    //   // std::cout << "data_solverRV_Pardiso \t" << v_solverRV_Pardiso << '\n';
    //   // std::cout << "data_directPardiso \t" << v_directPardiso << '\n';
    //
    //   data_solverRV_LLT.col(j) += v_solverRV_LLT/nbRepetitions;
    //   data_solverRV_LDLT.col(j) += v_solverRV_LDLT/nbRepetitions;
    //   data_solverRV_Pardiso.col(j) += v_solverRV_Pardiso/nbRepetitions;
    //   data_directPardiso.col(j) += v_directPardiso/nbRepetitions;
    // }
  }

  std::cout << "Done computing, now saving ..." << '\n';

  // std::cout << "data_solverRV_LLT \t" << std::endl <<  data_solverRV_LLT << '\n';
  // std::cout << "data_solverRV_LDLT \t" << std::endl <<  data_solverRV_LDLT << '\n';
  // std::cout << "data_solverRV_Pardiso \t" << std::endl <<  data_solverRV_Pardiso << '\n';
  // std::cout << "data_directPardiso \t" << std::endl <<  data_directPardiso << '\n';

  time_t t = time(NULL);
	tm* timePtr = localtime(&t);

  std::string dateEnd =  std::to_string(timePtr->tm_mon) + "." + std::to_string(timePtr->tm_mday) + "-" + std::to_string(timePtr->tm_hour) + ":" + std::to_string(timePtr->tm_min) + ":" + std::to_string(timePtr->tm_sec);

  saveToFile(data_solverRV_LLT, std::string("./timings_out/TEM_data_solverRV_LLT_" + dateEnd + ".m"));
  saveToFile(data_solverRV_LDLT, std::string("./timings_out/TEM_data_solverRV_LDLT_" + dateEnd + ".m"));
  saveToFile(data_solverRV_Pardiso, std::string("./timings_out/TEM_data_solverRV_Pardiso_" + dateEnd + ".m"));
  saveToFile(data_directPardiso, std::string("./timings_out/TEM_data_directPardiso_" + dateEnd + ".m"));
  saveToFile(nx_range, std::string("./timings_out/TEM_data_nx_range_" + dateEnd + ".m"));

  std::ofstream data_file;
  data_file.open ("./timings_out/data_information_"+ dateEnd +".txt");
  data_file << "Information on the run of timings.\n";
  data_file << "Started at " << dateStart << '\n';
  data_file << "Ended at   " << dateEnd << '\n';

  data_file << "\nnx_range : " << nx_range.transpose() << std::endl;
  data_file << nbRepetitions << " repetitions of each algorithm for averaging" << std::endl;
  data_file << "Algorithms ran on " << Eigen::nbThreads() << "OpenMP thread(s)";
  data_file.close();

  return 0;
}



void runAlgorithms(const Eigen::SparseMatrix<double>& R, const Eigen::SparseMatrix<double>& S,
  const Eigen::VectorXd& phi, const Eigen::VectorXd& psi, MatrixInfo& v_solverRV_LDLT,
  MatrixInfo& v_solverRV_LLT, MatrixInfo& v_solverRV_Pardiso,
  MatrixInfo& v_directPardiso)
{

  int n = R.rows();
  Eigen::VectorXcd u(n); u.setZero();
  Eigen::VectorXcd b(n);
  b.real() = phi; b.imag() = psi;

  //////////////////////////////////////////////////////////////////////////////
  ///                SolverRV_LDLT Object test                               ///
  //////////////////////////////////////////////////////////////////////////////
  double time_ref = omp_get_wtime();
  size_t start_RSS = getCurrentRSS();
  RealValuedSolverLDLT solverRV_LDLT(R, S);
  u = solverRV_LDLT.solve(b);

  // solverRV_LDLT.info(); // for now
  v_solverRV_LDLT(5) = omp_get_wtime() - time_ref;
  v_solverRV_LDLT.block(0,0,10,1) = solverRV_LDLT.information();
  v_solverRV_LDLT(8) = sqrt((R*u.real()-S*u.imag()-phi).squaredNorm() + (R*u.imag()+S*u.real()-psi).squaredNorm());
  // std::cout << "Vector information : " << v_solverRV_LDLT << '\n';
  v_solverRV_LDLT(10) = getCurrentRSS() - start_RSS;
  // std::cout << "RAM usage : " << getCurrentRSS() - start_RSS << '\n';
  std::cout << "v_solverRV_LDLT " << '\n' << v_solverRV_LDLT.transpose() << std::endl;
  u.setZero();

  //////////////////////////////////////////////////////////////////////////////
  ///                SolverRV_LLT Object test                                ///
  //////////////////////////////////////////////////////////////////////////////
  time_ref = omp_get_wtime();
  start_RSS = getCurrentRSS();
  RealValuedSolverLLT solverRV_LLT(R, S);
  u = solverRV_LLT.solve(b);

  // solverRV_LLT.info(); // for now
  v_solverRV_LDLT(5) = omp_get_wtime() - time_ref;
  v_solverRV_LLT.block(0,0,10,1) = solverRV_LLT.information();
  v_solverRV_LLT(8) = sqrt((R*u.real()-S*u.imag()-phi).squaredNorm() + (R*u.imag()+S*u.real()-psi).squaredNorm());
  // std::cout << "Vector information : " << v_solverRV_LLT << '\n';
  v_solverRV_LLT(10) = getCurrentRSS() - start_RSS;
  // std::cout << "RAM usage : " << getCurrentRSS() - start_RSS << '\n';
  std::cout << "v_solverRV_LLT " << '\n' << v_solverRV_LLT.transpose() << std::endl;
  u.setZero();

  //////////////////////////////////////////////////////////////////////////////
  ///                SolverRV_Pardiso Object test                            ///
  //////////////////////////////////////////////////////////////////////////////
  time_ref = omp_get_wtime();
  start_RSS = getCurrentRSS();
  RealValuedSolverPardiso solverRV_Pardiso(R, S);
  u = solverRV_Pardiso.solve(b);

  // solverRV_LDLT.info(); // for now
  v_solverRV_LDLT(5) = omp_get_wtime() - time_ref;
  v_solverRV_Pardiso.block(0,0,10,1) = solverRV_Pardiso.information();
  v_solverRV_Pardiso(8) = sqrt((R*u.real()-S*u.imag()-phi).squaredNorm() + (R*u.imag()+S*u.real()-psi).squaredNorm());
  // std::cout << "Vector information : " << v_solverRV_Pardiso << '\n';
  v_solverRV_Pardiso(10) = getCurrentRSS() - start_RSS;
  // std::cout << "RAM usage : " << getCurrentRSS() - start_RSS << '\n';
  std::cout << "v_solverRV_Pardiso " << '\n' << v_solverRV_Pardiso.transpose() << std::endl;
  u.setZero();

  //////////////////////////////////////////////////////////////////////////////
  ///                Pardiso direct solver test                              ///
  //////////////////////////////////////////////////////////////////////////////
  time_ref = omp_get_wtime();
  Eigen::SparseMatrix<std::complex<double>> A = R.cast<std::complex<double > >() + std::complex<double>(0.0,1.0) * S.cast<std::complex<double > >();
  clock_t overall_clock = clock();
  clock_t start_opt;
  start_RSS = getCurrentRSS();
  Eigen::PardisoLDLT<Eigen::SparseMatrix<std::complex<double>>, Eigen::Symmetric | Eigen::Upper> solverPardiso;
  start_opt = clock(); //

  solverPardiso.compute(A);
  v_directPardiso(0) = ( std::clock() - start_opt ) / (double) CLOCKS_PER_SEC; start_opt = clock();

  u = solverPardiso.solve(b);
  v_directPardiso(1) = ( std::clock() - start_opt ) / (double) CLOCKS_PER_SEC;

  v_directPardiso(5) = ( std::clock() - overall_clock ) / (double) CLOCKS_PER_SEC;
  v_solverRV_LDLT(5) = omp_get_wtime() - time_ref;

  v_directPardiso(8) = sqrt((R*u.real()-S*u.imag()-phi).squaredNorm() + (R*u.imag()+S*u.real()-psi).squaredNorm());
  // std::cout << "Vector information : " << v_solverRV_Pardiso << '\n';
  v_directPardiso(10) = getCurrentRSS() - start_RSS;
  // std::cout << "RAM usage : " << getCurrentRSS() - start_RSS << '\n';
  std::cout << "v_directPardiso " << '\n' << v_directPardiso.transpose() << std::endl;
  u.setZero();
}
