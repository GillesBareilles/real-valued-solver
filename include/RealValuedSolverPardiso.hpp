#include <iostream>
#include <vector>
#include <string>
#include <ctime>
#include <cmath>
#include <complex>

#define EIGEN_USE_MKL_ALL

#include <Eigen/PardisoSupport>

typedef Eigen::SparseMatrix<double> SpMat; // declares a column-major sparse matrix type of double

class RealValuedSolverPardiso
{
private:
  SpMat m_R;
  SpMat m_S;
  double m_CG_tol;        // relative tolerance for the CG stop condition
  double m_CG_maxit;
  double m_alpha;

  // double m_alphaHat;
  // double m_lambdaHat;

  bool m_initialized;
  bool m_adjustAlpha;

  Eigen::PardisoLDLT<Eigen::SparseMatrix<double>, Eigen::Symmetric | Eigen::Upper> m_solverBAlpha;

  Eigen::Matrix<double, 10, 1> m_information;

  void conjugateGradient(const Eigen::VectorXd& rhs, Eigen::VectorXd& x);

public:

  RealValuedSolverPardiso(const SpMat& R, const SpMat& S, double CG_tol, const double alpha);
  ~RealValuedSolverPardiso () {};

  void setTolerance(const double tol);
  void setMaxIt(const int maxIt);

  void factorize();
  void compute();

  Eigen::MatrixXcd solve(const Eigen::MatrixXcd& rhs);


  void info();
  Eigen::Matrix<double, 10, 1> information() const;
};


//////////// Implementation

RealValuedSolverPardiso::RealValuedSolverPardiso(const SpMat& R, const SpMat& S, double CG_tol = 1e-16, const double alpha = 1) : m_R(R), m_S(S), m_CG_tol(CG_tol), m_alpha(alpha), m_solverBAlpha()
{
  m_adjustAlpha = false;
  m_CG_maxit = 50;
  m_information.setZero();
  this->compute();
}

void RealValuedSolverPardiso::setTolerance(const double tol)
{
  m_CG_tol = tol;
}

void RealValuedSolverPardiso::setMaxIt(const int maxIt)
{
  m_CG_maxit = maxIt;
}

void RealValuedSolverPardiso::factorize()
{
  clock_t start_opt = clock();
  m_solverBAlpha.compute(m_R + m_alpha * m_S);
  m_information(0) = (( std::clock() - start_opt ) / (double) CLOCKS_PER_SEC);

  if (m_solverBAlpha.info() != Eigen::Success) m_information(9) = -1;
  else m_information(9) = 0;

  if (m_solverBAlpha.info() == Eigen::InvalidInput) {
    std::cout << "Facrtorize : Invalid Input" << '\n';
  } else if (m_solverBAlpha.info() == Eigen::NoConvergence) {
    std::cout << "Facrtorize : No Convergence" << '\n';
  } else if (m_solverBAlpha.info() == Eigen::NumericalIssue) {
    std::cout << "Facrtorize : NumericalIssue" << '\n';
  }
}

void RealValuedSolverPardiso::compute()
{
  factorize();
}

Eigen::MatrixXcd RealValuedSolverPardiso::solve(const Eigen::MatrixXcd& rhs)
{
  // Real Valued iterative method for complex symmetric linear systems

  // double lambda = -1;
  double alpha = m_alpha;

  // if (m_adjustAlpha) {
  //   // lambda_hat = max(eigs(R\S));   // TODO : implement max(eig)
  //   lambda = 1;
  //   alpha = lambda / (1 + sqrt(1 + pow(lambda,2)));
  //   std::cout << "lambda = " << lambda << std::endl;
  //   std::cout << "alpha = " << alpha << std::endl;
  // }

  Eigen::MatrixXd temp;

  // Step 1 : computing f
  clock_t start_opt = clock();
  temp = rhs.imag() - alpha * rhs.real();
  const Eigen::MatrixXd f = rhs.real() + m_S * m_solverBAlpha.solve(temp);
  m_information(1) = (( std::clock() - start_opt ) / (double) CLOCKS_PER_SEC);

  // Step 2 : computing x, solution of C_alpha x = f, with B_alpha as precond
  start_opt = clock();
  Eigen::MatrixXd x(rhs.rows(), rhs.cols());
  for (int i=0;i<rhs.cols();i++) {
    Eigen::VectorXd ttemp(rhs.rows());
    conjugateGradient(f.col(i), ttemp);
    x.col(i) = ttemp;
  }
  m_information(2) = (( std::clock() - start_opt ) / (double) CLOCKS_PER_SEC);


  // Step 3 : Computing z
  start_opt = clock();
  temp = alpha * rhs.real() - rhs.imag() + (1 + pow(alpha,2)) * m_S * x;
  Eigen::MatrixXd z = m_solverBAlpha.solve(temp);
  m_information(3) = (( std::clock() - start_opt ) / (double) CLOCKS_PER_SEC);


  // step 4 : Computing y = Im(u)
  start_opt = clock();
  Eigen::MatrixXd y = alpha * x - z;
  m_information(4) = (( std::clock() - start_opt ) / (double) CLOCKS_PER_SEC);

  Eigen::MatrixXcd u(rhs.rows(), rhs.cols());
  u.real() = x;
  u.imag() = y;

  //
  for (int i=0;i<5;++i) {
    m_information(5) += m_information(i);
  }
  return u;
}


void RealValuedSolverPardiso::info()
{
  std::cout << "------ Information :" << '\n';
  std::cout << "Factorisation of B_alpha  \t" << m_information(0) << '\n';
  std::cout << "1. Computing f (1 solve)  \t" << m_information(1) << '\n';
  std::cout << "2. Computing x (CG)       \t" << m_information(2) << '\n';
  std::cout << "   -> nb iterations (CG)  \t   " << m_information(6) << '\n';
  std::cout << "   -> relative error (CG) \t   " << m_information(7) << '\n';
  std::cout << "3. Computing z (1 solve)  \t" << m_information(3) << '\n';
  std::cout << "4. Computing y            \t" << m_information(4) << '\n';
  std::cout << "Total time               \t" << m_information(5) << '\n';
}



void RealValuedSolverPardiso::conjugateGradient(const Eigen::VectorXd& rhs, Eigen::VectorXd& x)
{
  using std::sqrt;
  using std::abs;

  int iters = -1;
  double tol_error = -1;

  double tol = m_CG_tol;
  int maxIters = m_CG_maxit ;

  int n = x.size();

  Eigen::VectorXd residual = m_S * x;
  residual = m_solverBAlpha.solve(residual);
  residual = rhs - ((m_R - m_alpha * m_S)* x + (1 + m_alpha * m_alpha) * m_S * residual);

  double rhsNorm2 = rhs.squaredNorm();
  if(rhsNorm2 == 0)
  {
    x.setZero();
    iters = 0;
    tol_error = 0;
    return;
  }
  double threshold = tol*tol*rhsNorm2;
  double residualNorm2 = residual.squaredNorm();
  if (residualNorm2 < threshold)
  {
    iters = 0;
    tol_error = sqrt(residualNorm2 / rhsNorm2);
    return;
  }

  Eigen::VectorXd p(n);
  p = m_solverBAlpha.solve(residual);      // initial search direction

  Eigen::VectorXd z(n), tmp(n);
  double absNew = abs(residual.dot(p));  // the square of the absolute value of r scaled by invM
  int i = 0;
  while(i < maxIters)
  {
    // Computing C_alpha * p
    tmp = m_S * p;
    tmp = m_solverBAlpha.solve(tmp);
    tmp = (m_R - m_alpha * m_S)* p + (1 + m_alpha * m_alpha) * m_S * tmp; // the bottleneck of the algorithm

    double alpha = absNew / p.dot(tmp);         // the amount we travel on dir
    x += alpha * p;                             // update solution
    residual -= alpha * tmp;                    // update residual

    residualNorm2 = residual.squaredNorm();
    if(residualNorm2 < threshold)
      break;

    z = m_solverBAlpha.solve(residual);                // approximately solve for "A z = residual"

    double absOld = absNew;
    absNew = abs(residual.dot(z));     // update the absolute value of r
    double beta = absNew / absOld;          // calculate the Gram-Schmidt value used to create the new search direction
    p = z + beta * p;                           // update search direction
    i++;
  }
  tol_error = sqrt(residualNorm2 / rhsNorm2);
  iters = i;

  m_information(6) = iters;
  m_information(7) = tol_error;
}

Eigen::Matrix<double, 10, 1> RealValuedSolverPardiso::information() const
{
  return m_information;
}
