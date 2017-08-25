#include <unistd.h>
#include <iostream>
#include <fstream>
#include <string>
#include <ctime>

// #include <QImage>
#include <vector>
#include <limits> // specified precision on cout
#include <Eigen/Sparse>

// #include "RealValuedSolver.hpp"
// #include "matrixFree.hpp"
// #include "CustomPreconditioner.hpp"

// Eigenvalues of sparse matrix
#include <GenEigsSolver.h>
#include <MatOp/SparseGenMatProd.h>
#include <SymEigsSolver.h>  // Also includes <MatOp/DenseSymMatProd.h>
#include <SymEigsShiftSolver.h>  // Also includes <MatOp/DenseSymShiftSolve.h>
#include <MatOp/SparseSymShiftSolve.h>

typedef Eigen::SparseMatrix<double> SpMat; // declares a column-major sparse matrix type of double
typedef Eigen::Triplet<double> T;
typedef std::numeric_limits< long double > dbl;

void saveToFile(const Eigen::VectorXd& x, std::string filename, bool doublePrecision = true);
void saveToFile(const Eigen::SparseMatrix<double>& x, std::string filename, bool doublePrecision = true);
void saveToFile(const Eigen::MatrixXd& x, std::string filename, bool doublePrecision = true);
// void saveAsBitmap(const Eigen::VectorXd& x, int n, const char* filename);
void buildProblem(std::vector<T>& coefficients, Eigen::VectorXd& b, int n);
void insertCoefficient(int id, int i, int j, double w, std::vector<T>& coeffs,
                       Eigen::VectorXd& b, const Eigen::VectorXd& boundary);
void buildRVProblem(std::vector<T>& coefficients_R, std::vector<T>& coefficients_S, Eigen::VectorXd& phi, Eigen::VectorXd& psi, int nx);
void poissonProblem();
void read_tem_matrices(std::vector<T>& coefficients_R, std::vector<T>& coefficients_S, Eigen::VectorXd& q, int n);
void read_matrix_MtxMarket(std::string file_name, SpMat& M);

void process_mem_usage(double& vm_usage, double& resident_set);

void info_matrix(const Eigen::SparseMatrix<double>& K);
// void compareMathods(const int nx, std::vector<double> &resultsTime, const int nbRepetition);
// void testsTiming();

// void RVAlgo(const SpMat& R, const SpMat& S, const Eigen::VectorXd& phi, const Eigen::VectorXd& psi, Eigen::VectorXd& x, Eigen::VectorXd& y, bool preconditionned = true, bool adjust_alpha = false);
