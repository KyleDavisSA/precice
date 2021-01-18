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
    /*
    if(firstIter){
      PRECICE_INFO("Using value preconditioner for first step.");
      std::vector<double> norms(_subVectorSizes.size(), 0.0);

      int offset = 0;
      for (size_t k = 0; k < _subVectorSizes.size(); k++) {
        Eigen::VectorXd part = Eigen::VectorXd::Zero(_subVectorSizes[k]);
        for (size_t i = 0; i < _subVectorSizes[k]; i++) {
          part(i) = oldValues(i + offset);
        }
        norms[k] = utils::MasterSlave::l2norm(part);
        offset += _subVectorSizes[k];
        PRECICE_ASSERT(norms[k] > 0.0);
      }

      offset = 0;
      for (size_t k = 0; k < _subVectorSizes.size(); k++) {
        for (size_t i = 0; i < _subVectorSizes[k]; i++) {
          _weights[i + offset]    = 1.0 / norms[k];
          _invWeights[i + offset] = norms[k];
        }
        offset += _subVectorSizes[k];
      }

      _requireNewQR  = true;
      firstIter = false;

    }else{
      */
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
    PRECICE_CHECK(not math::equals(sum, 0.0), "All residual sub-vectors in the residual-sum preconditioner are numerically zero. "
                                              "Your simulation probably got unstable, e.g. produces NAN values.");

    for (size_t k = 0; k < _subVectorSizes.size(); k++) {
      _residualSum[k] += norms[k] / sum;
      PRECICE_CHECK(not math::equals(_residualSum[k], 0.0), "A sub-vector in the residual-sum preconditioner became numerically zero. "
                                                            "Thus, the preconditioner is no longer stable. Please try the value preconditioner instead.");
    }

    offset = 0;
    Eigen::VectorXd normWeights;
    normWeights.resize(_subVectorSizes.size());
    for (size_t k = 0; k < _subVectorSizes.size(); k++) {
      for (size_t i = 0; i < _subVectorSizes[k]; i++) {
        _weights[i + offset]    = 1 / _residualSum[k];
        _invWeights[i + offset] = _residualSum[k];
      }
      normWeights[k] = 1 / _residualSum[k];
      PRECICE_INFO("Norm of weights: " << normWeights[k]);
      offset += _subVectorSizes[k];
      if (k == 0){
        if (normWeights[k] > maxWeight)
          maxWeight = normWeights[k];
        if (normWeights[k] < minWeight && normWeights[k] > 1000)
          minWeight = normWeights[k];
      }
      PRECICE_INFO("Norm of weights: Min " << minWeight << " and Max: " << maxWeight);
    }
    
    //PRECICE_INFO("Norm of weights: " << utils::MasterSlave::l2norm(normWeights));

    _requireNewQR = true;
    //}
  }else{

    for (size_t k = 0; k < _subVectorSizes.size(); k++) {
      _residualSum[k] = 0.0;
    }
  }
}

} // namespace impl
} // namespace acceleration
} // namespace precice
