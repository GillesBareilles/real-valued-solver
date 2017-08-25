#ifndef CUSTOM_PRECONDITIONERS_HPP
#define CUSTOM_PRECONDITIONERS_HPP

namespace Eigen {

/** \brief A preconditioner associated with a specific matrix B_alpha
  */

template <typename _Scalar>
class RVPreconditioner
{
    typedef _Scalar Scalar;
    typedef Matrix<Scalar,Dynamic,1> Vector;
  public:
    typedef typename Vector::StorageIndex StorageIndex;
    enum {
      ColsAtCompileTime = Dynamic,
      MaxColsAtCompileTime = Dynamic
    };

    RVPreconditioner() : m_isInitialized(false)
    {
      // std::cout << "Call RVPreconditioner()" << '\n';
    }

    template<typename MatType>
    explicit RVPreconditioner(const MatType& mat) : m_invdiag(mat.cols())
    {
      // std::cout << "Call to RVPreconditioner(const MatType& mat)" << '\n';
      compute(mat);
    }

    Index rows() const { return m_B_alpha.rows(); }
    Index cols() const { return m_B_alpha.cols(); }

    template<typename MatrixType>
    void setPrecond(const MatrixType& mat)
    {
      // std::cout << "Call to setPrecond" << '\n';
      m_B_alpha = mat;
    }

    template<typename MatType>
    RVPreconditioner& analyzePattern(const MatType& )
    {
      // std::cout << "Call to analyzePattern" << '\n';
      return *this;
    }

    template<typename MatType>
    RVPreconditioner& factorize(const MatType& mat)
    {
      // std::cout << "Call to fatorize" << '\n';
      return *this;
    }

    template<typename MatType>
    RVPreconditioner& compute(const MatType& mat)
    {
      // std::cout << "Call to compute !" << '\n';
      // m_cg.compute(m_B_alpha);
      m_ldlt.compute(m_B_alpha);
      m_isInitialized = true;
      return *this;
    }

    /** \internal */
    template<typename Rhs, typename Dest>
    void _solve_impl(const Rhs& b, Dest& x) const
    {
      // std::cout << "Call to _solve_impl" << '\n';
      // x = m_cg.solve(b);
      x = m_ldlt.solve(b);
      // std::cout << "#iterations:     " << m_cg.iterations()   << std::endl;
      // std::cout << "estimated error: " << m_cg.error()        << std::endl;
      // std::cout << x << std::endl;
    }

    template<typename Rhs>
    inline const Solve<RVPreconditioner, Rhs> // exit type
    solve(const MatrixBase<Rhs>& b) const
    {
      // std::cout << "Call to solve" << '\n';
      eigen_assert(m_isInitialized && "RVPreconditioner is not initialized.");
      eigen_assert(m_B_alpha.rows()==b.rows()
                && "RVPreconditioner::solve(): invalid number of rows of the right hand side matrix b");
      return Solve<RVPreconditioner, Rhs>(*this, b.derived());
    }

    ComputationInfo info() { return Success; }

  protected:
    Vector m_invdiag;
    bool m_isInitialized;
    Eigen::SparseMatrix<Scalar> m_B_alpha;
    // Eigen::ConjugateGradient<Eigen::SparseMatrix<Scalar>, Eigen::Lower|Eigen::Upper> m_cg;

    Eigen::SimplicialLDLT<Eigen::SparseMatrix<Scalar>> m_ldlt;
};

} // end namespace Eigen

#endif // CUSTOM_PRECONDITIONERS_HPP
