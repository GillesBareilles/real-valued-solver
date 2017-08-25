#include "utils.hpp"

void saveToFile(const Eigen::VectorXd& x, std::string filename, bool doublePrecision)
{
  std::fstream fs;
  fs.open (filename.c_str(), std::fstream::in | std::fstream::out | std::fstream::trunc);
  if (doublePrecision) fs.precision(dbl::max_digits10);
  else fs.precision(5);

  for (int i=0;i<x.size();i++) {
    fs << std::fixed <<  x(i) << std::endl;
  }

  fs.close();
}

void saveToFile(const Eigen::SparseMatrix<double>& A, std::string filename, bool doublePrecision)
{
  std::fstream fs;
  fs.open (filename.c_str(), std::fstream::in | std::fstream::out | std::fstream::trunc);
  if (doublePrecision) fs.precision(dbl::max_digits10);
  else fs.precision(5);

  for (int k=0; k < A.outerSize(); ++k)
  {
      for (Eigen::SparseMatrix<double>::InnerIterator it(A,k); it; ++it)
      {
          fs << 1+it.row() << " " << 1+it.col() << " " << std::fixed << it.value() << std::endl;
      }
  }
  fs.close();
}

void saveToFile(const Eigen::MatrixXd& A, std::string filename, bool doublePrecision)
{
  std::fstream fs;
  fs.open (filename.c_str(), std::fstream::in | std::fstream::out | std::fstream::trunc);
  if (doublePrecision) fs.precision(dbl::max_digits10);
  else fs.precision(5);

  for (int i=0; i < A.rows(); ++i)
  {
      for (int j=0; j < A.cols(); ++j)
      {
          fs << A(i,j) << " ";
      }
      fs << '\n';
  }
  fs.close();
}

// void saveAsBitmap(const Eigen::VectorXd& x, int n, const char* filename)
// {
//   Eigen::Array<unsigned char,Eigen::Dynamic,Eigen::Dynamic> bits = (x*255).cast<unsigned char>();
//   QImage img(bits.data(), n,n,QImage::Format_Indexed8);
//   img.setColorCount(256);
//   for(int i=0;i<256;i++) img.setColor(i,qRgb(i,i,i));
//   img.save(filename);
// }


void insertCoefficient(int id, int i, int j, double w, std::vector<T>& coeffs,
                       Eigen::VectorXd& b, const Eigen::VectorXd& boundary)
{
  int n = boundary.size();
  int id1 = i+j*n;

        if(i==-1 || i==n) b(id) -= w * boundary(j); // constrained coefficient
  else  if(j==-1 || j==n) b(id) -= w * boundary(i); // constrained coefficient
  else  coeffs.push_back(T(id,id1,w));              // unknown coefficient
}

void buildProblem(std::vector<T>& coefficients, Eigen::VectorXd& b, int n)
{
  b.setZero();
  Eigen::ArrayXd boundary = Eigen::ArrayXd::LinSpaced(n, 0,M_PI).sin().pow(2);
  for(int j=0; j<n; ++j)
  {
    for(int i=0; i<n; ++i)
    {
      int id = i+j*n;
      insertCoefficient(id, i-1,j, -1, coefficients, b, boundary);
      insertCoefficient(id, i+1,j, -1, coefficients, b, boundary);
      insertCoefficient(id, i,j-1, -1, coefficients, b, boundary);
      insertCoefficient(id, i,j+1, -1, coefficients, b, boundary);
      insertCoefficient(id, i,j,    4, coefficients, b, boundary);
    }
  }
}

void buildRVProblem(std::vector<T>& coefficients_R, std::vector<T>& coefficients_S, Eigen::VectorXd& phi, Eigen::VectorXd& psi, int nx)
{
  int n = phi.size();

  for(int i=0;i<n;i++)
  {
    coefficients_R.push_back(T(i,i,4.5));

    if ((i+nx-1)%nx != nx-1) coefficients_R.push_back(T(i,i-1,-1.125));
    if ((i+1)%nx !=0) coefficients_R.push_back(T(i,i+1,-1.125));
    if (i-nx>=0) coefficients_R.push_back(T(i,i-nx,-1.125));
    if (i+nx<n) coefficients_R.push_back(T(i,i+nx,-1.125));

    coefficients_S.push_back(T(i, i, 2));
    if (i-1>=0) coefficients_S.push_back(T(i, i-1, -1));
    if (i+1<n) coefficients_S.push_back(T(i, i+1, -1));
  }

  phi.setRandom(n,1);
  psi.setRandom(n,1);
}

void poissonProblem()
{
  int n = 100;  // size of the image
  int m = n*n;  // number of unknows (=number of pixels)

  // Assembly:
  std::vector<T> coefficients;            // list of non-zeros coefficients
  Eigen::VectorXd b(m);                   // the right hand side-vector resulting from the constraints
  buildProblem(coefficients, b, n);

  SpMat A(m,m);
  A.setFromTriplets(coefficients.begin(), coefficients.end());

  // Solving:
  Eigen::SimplicialCholesky<SpMat> chol(A);  // performs a Cholesky factorization of A
  Eigen::VectorXd x = chol.solve(b);         // use the factorization to solve for the given right hand side

  // Export the result to a file:
  // saveAsBitmap(x, n, "./out/img.bmp");
  saveToFile(x, std::string("./out/x.m"));
  std::cout << "2D pb done !"  << std::endl;
}

void read_matrix_MtxMarket(std::string file_name, Eigen::SparseMatrix<double>& M)
{
  std::ifstream matrix_file;
  matrix_file.open(file_name, std::ifstream::in);

  if(!matrix_file.is_open()) std::cout << "Error opening " << file_name << ".\n";

  std::string line;

  while (matrix_file.peek() == '%')
  {
    std::getline(matrix_file, line);
  }

  int n_row, n_col, nnz;
  matrix_file >> n_row >> n_col >> nnz;

  std::vector<T> coefficients;
  int i, j; double val;
  while (matrix_file >> i >> j >> val) {
    coefficients.push_back(T(i-1, j-1, val)); // 0 based indexing
    if (i != j) coefficients.push_back(T(j-1, i-1, val)); // building explicit symmetric matrix
  }
  matrix_file.close();

  M.resize(n_row, n_col);
  M.reserve(nnz);
  M.setFromTriplets(coefficients.begin(), coefficients.end());
}

void read_tem_matrices(std::vector<T>& coefficients_R, std::vector<T>& coefficients_S, Eigen::VectorXd& q, int n)
{
  SpMat R(n,n);  SpMat S(n,n);
  q.resize(n);
  int i, j;
  double value;

  std::ifstream matrixS; // -> S
  matrixS.open("./ressources/TEM" + std::to_string(n) + "_C.txt", std::ifstream::in);
  if(!matrixS.is_open()) std::cout << "Error opening " << "./ressources/TEM" << std::to_string(n) << "_C.txt" << ".\n";

  while (matrixS >> i) {
    matrixS >> j;
    matrixS >> value;
    // std::cout << i << " " << j << " " << value << '\n';
    coefficients_S.push_back(T(i-1, j-1, value));
  }
  matrixS.close();
  // std::cout << "coefficients_S : " << coefficients_S.size() << '\n';

  std::ifstream matrixR; // -> R
  matrixR.open("./ressources/TEM" + std::to_string(n) + "_M.txt", std::ifstream::in);
  if(!matrixR.is_open()) std::cout << "Error opening " << "./ressources/TEM" << std::to_string(n) << "_M.txt" << ".\n";
  while (matrixR >> i) {
    matrixR >> j;
    matrixR >> value;
    // std::cout << i << " " << j << " " << value << '\n';
    coefficients_R.push_back(T(i-1, j-1, value));
  }
  matrixR.close();
  // std::cout << "coefficients_R : " << coefficients_R.size() << '\n';

  std::ifstream vectorq;
  vectorq.open("./ressources/TEM" + std::to_string(n) + "_q.txt", std::ifstream::in);
  if(!vectorq.is_open()) std::cout << "Error opening " << "./ressources/TEM" << std::to_string(n) << "_q.txt" << ".\n";
  i = 0;
  while (vectorq >> value) {
    q(i) = value;
    i+=1;
  }
  vectorq.close();
  // std::cout << "vectorq : " << q.size() << '\n' << '\n';
}


void info_matrix(const Eigen::SparseMatrix<double>& K)
{
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
  std::cout << "Norm of antisymmetric part : " << (K-Ktr).norm() << '\n';
  std::cout << "Norme of matrix : " << K.norm() << '\n' << std::endl;
}
//////////////////////////////////////////////////////////////////////////////
//
// process_mem_usage(double &, double &) - takes two doubles by reference,
// attempts to read the system-dependent data for a process' virtual memory
// size and resident set size, and return the results in KB.
//
// On failure, returns 0.0, 0.0

/*
 * Author:  David Robert Nadeau
 * Site:    http://NadeauSoftware.com/
 * License: Creative Commons Attribution 3.0 Unported License
 *          http://creativecommons.org/licenses/by/3.0/deed.en_US
 */


// void compareMathods(const int nx, std::vector<double> &resultsTime, const int nbRepetition)
// {
//   std::cout << "--------------------------------------------------------------------------------" << '\n';
//   std::cout << "--------------- BiCGSTAB vc. RV precond vs. RV wrapped -- nx = " << nx << "----------------" << '\n';
//   resultsTime.clear();
//
//   int n = pow(nx,2);
//   Eigen::VectorXd phi(n);
//   Eigen::VectorXd psi(n);
//   Eigen::VectorXd x(n);
//   Eigen::VectorXd y(n);
//   SpMat R(n,n);
//   SpMat S(n,n);
//
//   std::vector<T> coefficients_R, coefficients_S;
//   buildRVProblem(coefficients_R, coefficients_S, phi,  psi, nx);
//
//   R.setFromTriplets(coefficients_R.begin(), coefficients_R.end());
//   S.setFromTriplets(coefficients_S.begin(), coefficients_S.end());
//
//   Eigen::SparseMatrix<std::complex<double > > A(n,n);
//   Eigen::VectorXcd u(n);
//   Eigen::VectorXcd b(n);
//
//   A = R.cast<std::complex<double > >() + std::complex<double>(0.0,1.0) * S.cast<std::complex<double > >();
//   b = phi.cast<std::complex<double > >() + std::complex<double>(0.0,1.0) * psi.cast<std::complex<double > >();
//
//   clock_t start_opt;
//   //////////////////////////////////////////////////////////////////////////////
//   ///                      Direct Complex Method                             ///
//   //////////////////////////////////////////////////////////////////////////////
//
//   for (int i=0;i<nbRepetition;++i)
//   {
//     std::cout << "Direct complex solve : it " << i << '\n';
//     start_opt = clock();
//     Eigen::BiCGSTAB<Eigen::SparseMatrix<std::complex<double>> > solver;
//     solver.compute(A);
//     u = solver.solve(b);
//
//     resultsTime[0] += ( std::clock() - start_opt ) / (double) CLOCKS_PER_SEC;
//     std::cout << "Direct solver, BiCGSTAB:     " << ( std::clock() - start_opt ) / (double) CLOCKS_PER_SEC << " s" << std::endl;
//     std::cout << "Résidual :  " << (A*u-b).norm() << std::endl << std::endl;
//   }
//
//   //////////////////////////////////////////////////////////////////////////////
//   ///                Actually preconditionned Method                         ///
//   //////////////////////////////////////////////////////////////////////////////
//   double alpha = 1;
//
//   for (int i=0;i<nbRepetition;++i)
//   {
//     std::cout << "RV precond solve : it " << i << '\n';
//     clock_t timeRV;
//     timeRV = clock();
//
//     // Setting up the intermidiary matrices and vectors
//     SpMat B_alpha = R + alpha*S; // TODO : impliciter...
//
//     // Setting the solver related to B_alpha
//     start_opt = clock();
//     Eigen::SimplicialLDLT<SpMat > solver_B_alpha;
//     solver_B_alpha.compute(B_alpha);
//     std::cout << "Init : solver_B_alpha " << ( std::clock() - start_opt ) / (double) CLOCKS_PER_SEC << " s" << std::endl;
//
//     start_opt = clock();
//     MatrixReplacement C_alpha;
//     C_alpha.attachMyMatrices(R, S, alpha);
//     std::cout << "Init - init MatrixReplacement C_alpha " << ( std::clock() - start_opt ) / (double) CLOCKS_PER_SEC << " s" << std::endl;
//
//
//     // Step 1 : computing f
//     start_opt = clock();
//     Eigen::VectorXd f = phi + S * solver_B_alpha.solve(psi - alpha * phi);
//     std::cout << "Step 1 : calcul f     " << ( std::clock() - start_opt ) / (double) CLOCKS_PER_SEC << " s" << std::endl;
//
//
//     // Step 2 : computing x, solution of C_alpha x = f, with B_alpha as precond
//     start_opt = clock();
//     Eigen::ConjugateGradient<MatrixReplacement, Eigen::Lower|Eigen::Upper, Eigen::RVPreconditioner<double>> cg;
//     cg.preconditioner().setPrecond(B_alpha);
//     cg.compute(C_alpha);
//     std::cout << "Step 2 - init CG solver  " << ( std::clock() - start_opt ) / (double) CLOCKS_PER_SEC << " s" << std::endl;
//     start_opt = clock();
//     x = cg.solve(f);
//     std::cout << "Step 2 - calcul x CG   " << ( std::clock() - start_opt ) / (double) CLOCKS_PER_SEC << " s" << std::endl;
//
//
//     // Step 3 : Computing z
//     start_opt = clock();
//     Eigen::VectorXd z = solver_B_alpha.solve(alpha * phi - psi + (1 + pow(alpha,2)) * S * x);
//     std::cout << "Step 3 : calcul z     " << ( std::clock() - start_opt ) / (double) CLOCKS_PER_SEC << " s" << std::endl;
//
//     // step 4 : Computing y = Im(u)
//     start_opt = clock();
//     y = alpha * x - z;
//     std::cout << "Step 4 : calcul y     " << ( std::clock() - start_opt ) / (double) CLOCKS_PER_SEC << " s" << std::endl << std::endl;
//
//     // Results :
//     resultsTime[1] = ( std::clock() - timeRV ) / (double) CLOCKS_PER_SEC;
//     std::cout << "RV algo, CG precond    :     " << ( std::clock() - timeRV ) / (double) CLOCKS_PER_SEC << " s" << std::endl;
//     u = x.cast<std::complex<double > >() + std::complex<double>(0.0,1.0) * y.cast<std::complex<double > >();
//     std::cout << "Residual :  " << (A*u-b).norm() << std::endl << std::endl;
//   }
//
//
//   //////////////////////////////////////////////////////////////////////////////
//   ///                SolverRealValued Object test                            ///
//   //////////////////////////////////////////////////////////////////////////////
//
//   for (int i=0;i<nbRepetition;++i)
//   {
//     std::cout << "Direct complex solve : it " << i << '\n';
//
//     start_opt = clock();
//     RealValuedSolver solverRV(R, S, alpha);
//     solverRV.compute();
//     u = solverRV.solve(b);
//
//     resultsTime[2] = ( std::clock() - start_opt ) / (double) CLOCKS_PER_SEC;
//     std::cout << "Direct solver, wrapped :     " << ( std::clock() - start_opt ) / (double) CLOCKS_PER_SEC << " s" << std::endl;
//     std::cout << "Résidual :  " << (A*u-b).norm() << std::endl << std::endl;
//   }
// }

// void testsTiming()
// {
//   Eigen::VectorXd nx_range = Eigen::VectorXd::LinSpaced(40,1,log(300));
//   nx_range = nx_range.array().exp().ceil();
//
//   // nx_range = Eigen::VectorXd::LinSpaced(7,1,1);
//
//   std::cout << "Evaluating for nx spanning : ";
//   for (int i=0;i<nx_range.size();i++) {
//     std::cout << nx_range(i) << ' ';
//   }
//   std::cout << '\n';
//
//
//   std::vector<double> resultsTime(3);
//   Eigen::MatrixXd results(3, nx_range.size());
//
//   int nbRepetitions = 5;
//   for (int i=0;i<nx_range.size();i++) {
//     // compareMathods(nx_range(i), resultsTime, nbRepetitions);
//     results(0,i) = resultsTime[0]/nbRepetitions;
//     results(1,i) = resultsTime[1]/nbRepetitions;
//     results(2,i) = resultsTime[2]/nbRepetitions;
//   }
//
//   std::cout << "Done computing, now saving ..." << '\n';
//
//   saveToFile(results, std::string("./out/results.m"), false);
//   saveToFile(nx_range, std::string("./out/nx_range.m"), false);
// }
