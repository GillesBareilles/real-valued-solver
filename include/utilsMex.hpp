#include <mex.h>
#include <Eigen/Sparse>
#include <vector>
#include <iostream>

typedef Eigen::Triplet<double> T;

void readSparse(const mxArray *prhs, Eigen::SparseMatrix<double> &mat);

void readComplexArray(const mxArray *prhs, std::vector<std::complex<double> > &vec);
void readRealArray(const mxArray *prhs, std::vector<double> &vec);

void readComplexVector(const mxArray *prhs, Eigen::VectorXcd &vec);
void readRealVector(const mxArray *prhs, Eigen::VectorXd &vec);
void readComplexMatrix(const mxArray *prhs, Eigen::MatrixXcd &mat);
void readRealMatrix(const mxArray *prhs, Eigen::MatrixXd &mat);

mxArray * writeRealArray(const std::vector<double> &v);
mxArray * writeComplexArray(const std::vector<std::complex<double> > &v);

mxArray * writeRealVector(const Eigen::VectorXd &vec);
mxArray * writeComplexVector(const Eigen::VectorXcd &vec);
mxArray * writeComplexMatrix(const Eigen::MatrixXcd &mat);
mxArray * writeRealMatrix(const Eigen::MatrixXd &mat);

void setSparseIdentity(Eigen::SparseMatrix<double>& mat, const int& nb_row);



void readSparse(const mxArray *prhs, Eigen::SparseMatrix<double> &mat)
{
  /* Convert matlab sparse matrix to eigen's sparse matrix Eigen::SparseMatrix<double>
  Compute the triplets of coordonates of values and values, then build the sparse matrix.

  IMPORTANT : The best way to work with matlab's input matrix would be to
  directly map the buffers as an Eigen sparse matrix. However, it seems eigen
  won't accept size_t index, for size_t is **unsigned** long int.
  */

  if (!(mxIsDouble(prhs))){
      mexErrMsgIdAndTxt( "MATLAB:readSparse:inputNotDouble",
              "Input argument must be of type double.");
  }

  if (mxGetNumberOfDimensions(prhs) != 2){
      mexErrMsgIdAndTxt( "MATLAB:readSparse:inputNot2D",
              "Input argument must be two dimensional\n");
  }

  int m = static_cast<int>(mxGetM(prhs));
  int n = static_cast<int>(mxGetN(prhs));
  int nnz = static_cast<int>(mxGetNzmax(prhs));
  size_t *row_ptr = (mxGetIr(prhs));
  size_t *col_index = (mxGetJc(prhs));
  double *values = mxGetPr(prhs);

  // Convert the size_t arrays to int arrays:
  std::vector<int> outerIndex;
  std::vector<int> innerIndex;
  for (int i=0; i<nnz;++i) innerIndex.push_back(static_cast<int>(row_ptr[i]));
  for (int i=0; i<m;++i) outerIndex.push_back(static_cast<int>(col_index[i]));


  // Computing the eigen's triplets from matlab's compressed storage:
  std::vector<T> coefficients;
  int k=0;
  int j=0;
  while(k<nnz) {
    while((k<outerIndex[j+1])&&(j<n-1)) {
      coefficients.push_back(T(innerIndex[k],j,values[k]));
      k++;
    }
    if (j==n-1) {
      while(k<nnz) {
        coefficients.push_back(T(innerIndex[k],j,values[k]));
        k++;
      }
    }
    j++;
  }

  mat.resize(m,n);
  mat.setFromTriplets(coefficients.begin(), coefficients.end());
  mat.makeCompressed();
}



// Building a complex array from matlab buffers
void readComplexArray(const mxArray *prhs, std::vector<std::complex<double> > &vec)
{
  /* Check data type of input argument  */
  if (mxGetNumberOfDimensions(prhs) != 2){
      mexErrMsgIdAndTxt( "MATLAB:readSparse:inputNot1D",
              "readComplexArray : Input argument must be one dimensional\n");
  }

  /* Check that both inputs are complex*/
  if( !mxIsComplex(prhs) )
  mexErrMsgIdAndTxt( "MATLAB:convec:inputsNotComplex",
  "Input argument must be complex.\n");

  if( mxGetN(prhs) != 1 )
      mexErrMsgIdAndTxt( "MATLAB:convec:inputNotVector",
              "Input must be column vector.");

  /* get the length the input vector */
  int n = mxGetM(prhs);

  /* get pointers to the real and imaginary parts of the inputs */
  double *xr = mxGetPr(prhs);
  double *xi = mxGetPi(prhs);

  for (int i=0;i<n;++i) vec.push_back(std::complex<double>(xr[i], xi[i]));
}


// Building a real array from matlab buffers
void readRealArray(const mxArray *prhs, std::vector<double> &vec)
{
  /* Check data type of input argument  */
  if (mxGetNumberOfDimensions(prhs) != 2){
    mexErrMsgIdAndTxt( "MATLAB:readSparse:inputNot1D",
    "readRealArray : Input argument must be one dimensional\n");
  }

  /* Check that both inputs are real*/
  if( !mxIsNumeric(prhs) )
  mexErrMsgIdAndTxt( "MATLAB:convec:inputsNotComplex",
  "Input argument must be complex.\n");

  if( mxGetN(prhs) != 1 )
  mexErrMsgIdAndTxt( "MATLAB:readRealArray:inputNotVector",
  "Input must be column vector.");

  /* get the length and pointer to input vector */
  int n = mxGetM(prhs);
  // mexPrintf("n : %i", mxGetN(prhs));
  // mexPrintf("m : %i", mxGetM(prhs));
  double *xr = mxGetPr(prhs);

  for (int i=0;i<n;++i) vec.push_back(xr[i]);
}


// Building a Eigen::VectorXcd from matlab buffers
void readComplexVector(const mxArray *prhs, Eigen::VectorXcd &vec)
{
  /* Check data type of input argument  */
  if (mxGetNumberOfDimensions(prhs) != 2){
      mexErrMsgIdAndTxt( "MATLAB:readSparse:inputNot1D",
              "readRealVector : Input argument must be one dimensional\n");
  }

  /* Check that both inputs are real*/
  if( !mxIsComplex(prhs) )
  mexErrMsgIdAndTxt( "MATLAB:convec:inputsNotComplex",
  "Input argument must be complex.\n");

  if( mxGetN(prhs) != 1 )
      mexErrMsgIdAndTxt( "MATLAB:readRealArray:inputNotVector",
              "Input must be column vector.");

  /* get the length and pointers to input vector */
  int n = mxGetM(prhs);
  double *xr = mxGetPr(prhs);
  double *xi = mxGetPi(prhs);

  for (int i=0;i<n;++i) vec(i) = std::complex<double>(xr[i], xi[i]);
}


// Building a Eigen::VectorXd from matlab buffers
void readRealVector(const mxArray *prhs, Eigen::VectorXd &vec)
{
  /* Check data type of input argument  */
  if (mxGetNumberOfDimensions(prhs) != 2){
      mexErrMsgIdAndTxt( "MATLAB:readSparse:inputNot1D",
              "readRealVector : Input argument must be one dimensional\n");
  }

  /* Check that both inputs are real*/
  if( !mxIsNumeric(prhs) )
  mexErrMsgIdAndTxt( "MATLAB:convec:inputsNotComplex",
  "Input argument must be complex.\n");

  if( mxGetN(prhs) != 1 )
      mexErrMsgIdAndTxt( "MATLAB:readRealArray:inputNotVector",
              "Input must be column vector.");

  /* get the length and pointer to input vector */
  int n = mxGetM(prhs);
  double *xr = mxGetPr(prhs);

  vec.resize(n);

  for (int i=0;i<n;++i) {
    vec(i) = xr[i];
  }
}

// Building a Eigen::MatrixXcd from matlab buffers
void readComplexMatrix(const mxArray *prhs, Eigen::MatrixXcd &mat)
{
  /* Check data type of input argument  */
  if (mxGetNumberOfDimensions(prhs) != 2){
      mexErrMsgIdAndTxt( "MATLAB:readSparse:inputNot1D",
              "readRealVector : Input argument must be one dimensional\n");
  }

  /* Check that both inputs are real*/
  if( !mxIsComplex(prhs) )
  mexErrMsgIdAndTxt( "MATLAB:convec:inputsNotComplex",
  "Input argument must be complex.\n");

  /* get the length and pointer to input vector */
  int m = mxGetM(prhs);                               // number of rows
  int n = mxGetN(prhs);                               // number of columns
  double *pr = mxGetPr(prhs);
  double *pi = mxGetPi(prhs);

  mat.resize(m,n);

  for (int j=0;j<mat.cols();++j) {
    for (int i=0;i<mat.rows();++i && ++pr && ++pi) {
      mat(i,j) = std::complex<double> (*pr, *pi);
    }
  }
}

// Building a Eigen::MatrixXd from matlab buffers
void readRealMatrix(const mxArray *prhs, Eigen::MatrixXd &mat)
{
  /* Check data type of input argument  */
  if (mxGetNumberOfDimensions(prhs) != 2){
      mexErrMsgIdAndTxt( "MATLAB:readSparse:inputNot1D",
              "readRealVector : Input argument must be one dimensional\n");
  }

  /* Check that both inputs are real*/
  if( !mxIsNumeric(prhs) )
  mexErrMsgIdAndTxt( "MATLAB:convec:inputsNotComplex",
  "Input argument must be complex.\n");

  /* get the length and pointer to input vector */
  int m = mxGetM(prhs);                               // number of rows
  int n = mxGetN(prhs);                               // number of columns
  double *pr = mxGetPr(prhs);

  mat.resize(m,n);

  for (int j=0;j<mat.cols();++j) {
    for (int i=0;i<mat.rows();++i && ++pr) {
      mat(i,j) = *pr;
    }
  }
}


mxArray * writeRealArray(const std::vector<double>& v)
{
    mxArray * mx = mxCreateDoubleMatrix(1,v.size(), mxREAL);
    std::copy(v.begin(), v.end(), mxGetPr(mx));
    return mx;
}


mxArray * writeComplexArray(const std::vector<std::complex<double> >& v)
{
  mxArray * mx = mxCreateDoubleMatrix(1, v.size(), mxCOMPLEX);
  double* pr = mxGetPr(mx);
  double* pi = mxGetPi(mx);
  for (std::complex<double> c:v) {
    *pr = c.real();
    *pi = c.imag();
    ++pr;
    ++pi;
  }
  return mx;
}

mxArray * writeRealVector(const Eigen::VectorXd &vec)
{
    mxArray * mx = mxCreateDoubleMatrix(1,vec.size(), mxREAL);
    double* pr = mxGetPr(mx);
    for (int i=0;i<vec.size();++i && ++pr) {
      *pr = vec(i);
      // mexPrintf("%g\n", vec(i));
    }

    return mx;
}


mxArray * writeComplexVector(const Eigen::VectorXcd &vec)
{
    mxArray * mx = mxCreateDoubleMatrix(vec.size(),1, mxCOMPLEX);
    double* pr = mxGetPr(mx);
    double* pi = mxGetPi(mx);
    for (int i=0;i<vec.size();++i && ++pr && ++pi) {
      *pr = vec(i).real();
      *pi = vec(i).imag();
    }
    return mx;
}
mxArray * writeComplexMatrix(const Eigen::MatrixXcd &mat)
{
    mxArray * mx = mxCreateDoubleMatrix(mat.rows(),mat.cols(), mxCOMPLEX);
    double* pr = mxGetPr(mx);
    double* pi = mxGetPi(mx);

    for (int j=0;j<mat.cols();++j) {
      for (int i=0;i<mat.rows();++i && ++pr && ++pi) {
        *pr = mat(i,j).real();
        *pi = mat(i,j).imag();
      }
    }    return mx;
}
mxArray * writeRealMatrix(const Eigen::MatrixXd &mat)
{
  mxArray * mx = mxCreateDoubleMatrix(mat.rows(),mat.cols(), mxREAL);
  double* pr = mxGetPr(mx);

  for (int j=0;j<mat.cols();++j) {
    for (int i=0;i<mat.rows();++i && ++pr) {
      *pr = mat(i,j);
    }
  }
  return mx;
}



void setSparseIdentity(Eigen::SparseMatrix<double>& mat, const int& nb_row)
{
  std::vector<T> coefficients;
  for (int i=0;i<nb_row;i++) {
    coefficients.push_back(T(i,i,1));
  }
  mat.resize(nb_row, nb_row);
  mat.setFromTriplets(coefficients.begin(), coefficients.end());
  mat.makeCompressed();
}
