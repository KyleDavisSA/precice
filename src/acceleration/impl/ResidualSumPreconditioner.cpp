#include "acceleration/impl/ResidualSumPreconditioner.hpp"
#include <algorithm>
#include <math.h>
#include "logging/LogMacros.hpp"
#include "math/differences.hpp"
#include "utils/MasterSlave.hpp"
#include "utils/assertion.hpp"

namespace precice {
namespace acceleration {
namespace impl {

ResidualSumPreconditioner::ResidualSumPreconditioner(
    int maxNonConstTimesteps)
    : Preconditioner(maxNonConstTimesteps)
{
}

void ResidualSumPreconditioner::initialize(std::vector<size_t> &svs)
{
  PRECICE_TRACE();
  Preconditioner::initialize(svs);

  _residualSum.resize(_subVectorSizes.size(), 0.0);
}

void ResidualSumPreconditioner::_update_(bool                   timestepComplete,
                                         const Eigen::VectorXd &oldValues,
                                         const Eigen::VectorXd &res)
{
  if (not timestepComplete) {
    std::vector<double> norms(_subVectorSizes.size(), 0.0);

    double sum = 0.0;

    int offset = 0;
    for (size_t k = 0; k < _subVectorSizes.size(); k++) {
      Eigen::VectorXd part = Eigen::VectorXd::Zero(_subVectorSizes[k]);
      for (size_t i = 0; i < _subVectorSizes[k]; i++) {
        part(i) = res(i + offset);
      }
      norms[k] = utils::MasterSlave::dot(part, part);
      sum += norms[k];
      offset += _subVectorSizes[k];
      norms[k] = std::sqrt(norms[k]);
    }
    sum = std::sqrt(sum);
    //PRECICE_CHECK(not math::equals(sum, 0.0), "All residual sub-vectors in the residual-sum preconditioner are numerically zero ( sum = " << sum <<
    //                                          "). This indicates that the data values exchanged between two succesive iterations did not change."
    //                                          " Your simulation probably got unstable, e.g. produces NAN values. Please check the data values exchanged" 
    //                                          " between the solvers.");
    if (math::equals(sum, 0.0)) {
      PRECICE_WARN("All residual sub-vectors in the residual-sum preconditioner are numerically zero ( sum = " << sum <<
                                              "). This indicates that the data values exchanged between two succesive iterations did not change."
                                              " The simulation may be unstable, e.g. produces NAN values. Please check the data values exchanged" 
                                              " between the solvers is not identical between iterations.");
    }

    for (size_t k = 0; k < _subVectorSizes.size(); k++) {
      _residualSum[k] += norms[k] / sum;
      //PRECICE_CHECK(not math::equals(_residualSum[k], 0.0), "A sub-vector in the residual-sum preconditioner became numerically zero ( residualSum = " << _residualSum[k] <<
      //                                                      ") . If this occured during the first iteration, check that the initial-relaxation factor is not equal to 1.0"
      //                                                      " if the coupling values of one solver is zero in the first iterations. "
      //                                                      " . Please try freezing the preconditioner weights or the value preconditioner instead.");
      if (math::equals(_residualSum[k], 0.0)) {
        PRECICE_WARN("A sub-vector in the residual-sum preconditioner became numerically zero ( residualSum = " << _residualSum[k] <<
                                                            ") . If this occured directly after the second iteration, check that the initial-relaxation"
                                                            " factor is not equal to 1.0 if the coupling values of one solver"
                                                            " is zero in the first iteration."
                                                            " Please try freezing the preconditioner weights when using residual-sum"
                                                            " or use the value preconditioner instead.");
      }
    }

    offset = 0;
    for (size_t k = 0; k < _subVectorSizes.size(); k++) {
      if (!math::equals(_residualSum[k], 0.0)){
      for (size_t i = 0; i < _subVectorSizes[k]; i++) {
          _weights[i + offset]    = 1 / _residualSum[k];
          _invWeights[i + offset] = _residualSum[k];
        }
        PRECICE_DEBUG("preconditioner weight[" << k << "] = " << 1 / _residualSum[k]);
      }
      offset += _subVectorSizes[k];
    }

    _requireNewQR = true;
  } else {
    for (size_t k = 0; k < _subVectorSizes.size(); k++) {
      _residualSum[k] = 0.0;
    }
  }
}

} // namespace impl
} // namespace acceleration
} // namespace precice
