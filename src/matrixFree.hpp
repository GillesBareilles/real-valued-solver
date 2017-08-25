#include <iostream>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/IterativeLinearSolvers>


typedef Eigen::Triplet<double> T;
typedef Eigen::SparseMatrix<double> SpMat; // declares a column-major sparse matrix type of double


class MatrixReplacement;
using Eigen::SparseMatrix;

namespace Eigen {
namespace internal {
  // MatrixReplacement looks-like a SparseMatrix, so let's inherits its traits:
  template<>
  struct traits<MatrixReplacement> :  public Eigen::internal::traits<Eigen::SparseMatrix<double> >
  {};
}
}

// Example of a matrix-free wrapper from a user type to Eigen's compatible type
// For the sake of simplicity, this example simply wrap a Eigen::SparseMatrix.
class MatrixReplacement : public Eigen::EigenBase<MatrixReplacement> {
public:
  // Required typedefs, constants, and method:
  typedef double Scalar;
  typedef double RealScalar;
  typedef int StorageIndex;
  enum {
    ColsAtCompileTime = Eigen::Dynamic,
    MaxColsAtCompileTime = Eigen::Dynamic,
    IsRowMajor = false
  };

  Index rows() const { return mp_R->rows(); }
  Index cols() const { return mp_R->cols(); }

  template<typename Rhs>
  Eigen::Product<MatrixReplacement,Rhs,Eigen::AliasFreeProduct> operator*(const Eigen::MatrixBase<Rhs>& x) const {
    return Eigen::Product<MatrixReplacement,Rhs,Eigen::AliasFreeProduct>(*this, x.derived());
  }

  // Custom API:
  MatrixReplacement() : mp_R(0), mp_S(0), m_alpha(0) {}

  void attachMyMatrix(const SparseMatrix<double> &mat) {
    mp_R = &mat;
  }

  void attachMyMatrices(const SparseMatrix<double> &mat_R, const SparseMatrix<double> &mat_S, double alpha) {
    mp_R = &mat_R;
    mp_S = &mat_S;
    m_alpha = alpha;
  }

  const SparseMatrix<double> my_R() const { return *mp_R; }
  const SparseMatrix<double> my_S() const { return *mp_S; }
  double my_alpha() const { return m_alpha; }

private:
  const SparseMatrix<double> *mp_R;
  const SparseMatrix<double> *mp_S;
  double m_alpha;
};


// Implementation of MatrixReplacement * Eigen::DenseVector though a specialization of internal::generic_product_impl:
namespace Eigen {
namespace internal {

  template<typename Rhs>
  struct generic_product_impl<MatrixReplacement, Rhs, SparseShape, DenseShape, GemvProduct> // GEMV stands for matrix-vector
  : generic_product_impl_base<MatrixReplacement,Rhs,generic_product_impl<MatrixReplacement,Rhs> >
  {
    typedef typename Product<MatrixReplacement,Rhs>::Scalar Scalar;

    template<typename Dest>
    static void scaleAndAddTo(Dest& dst, const MatrixReplacement& lhs, const Rhs& rhs, const Scalar& alpha)
    {
      // This method should implement "dst += alpha * lhs * rhs" inplace,
      // however, for iterative solvers, alpha is always equal to 1, so let's not bother about it.
      assert(alpha==Scalar(1) && "scaling is not implemented");

      // Here we could simply call dst.noalias() += lhs.my_matrix() * rhs,
      // but let's do something fancier (and less efficient):
      dst.noalias() += (lhs.my_R() - lhs.my_alpha() * lhs.my_S())* rhs;

      SpMat B_alpha = lhs.my_R() + lhs.my_alpha() * lhs.my_S();
      Eigen::SimplicialLDLT<SpMat > solver_B_alpha;
      solver_B_alpha.compute(B_alpha);

      Dest tmp = lhs.my_S()*rhs;
      tmp = solver_B_alpha.solve(tmp);
      dst.noalias() += (1+lhs.my_alpha() * lhs.my_alpha()) * lhs.my_S()*tmp;
    }
  };

}
}
