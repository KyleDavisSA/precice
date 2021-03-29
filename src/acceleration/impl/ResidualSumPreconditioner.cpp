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
  _setWeights.resize(_subVectorSizes.size(), 0.0);
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
    PRECICE_INFO("Sum of res-sum preconditioner: " << sum);
    if (math::equals(sum, 0.0)) {
      PRECICE_WARN("All residual sub-vectors in the residual-sum preconditioner are numerically zero ( sum = " << sum << "). This indicates that the data values exchanged between two succesive iterations did not change."
                                                                                                                         " The simulation may be unstable, e.g. produces NAN values. Please check the data values exchanged"
                                                                                                                         " between the solvers is not identical between iterations. The preconditioner scaling factors were"
                                                                                                                         " not updated in this iteration and the scaling factors determined in the previous iteration were used.");
      sum = 1.0;
    }

    for (size_t k = 0; k < _subVectorSizes.size(); k++) {
      _residualSum[k] += norms[k] / sum;
      PRECICE_INFO("Norm of res-sum: " << norms[k]);
      PRECICE_INFO("residualSum of res-sum: " << _residualSum[k]);
      if (math::equals(_residualSum[k], 0.0)) {
        PRECICE_WARN("A sub-vector in the residual-sum preconditioner became numerically zero ( sub-vector = " << _residualSum[k] << "). If this occured in the second iteration and the initial-relaxation factor is equal to 1.0,"
                                                                                                                                     " check if the coupling data values of one solver is zero in the first iteration."
                                                                                                                                     " The preconditioner scaling factors were not updated for this iteration and the scaling factors"
                                                                                                                                     " determined in the previous iteration were used.");
      }
    }

    offset = 0;
    int resetWeight = 0;
    normWeights.resize(_subVectorSizes.size());
    // Chech if the new scaling weights are more than 1 order of magnitude from the previous weights
    for (size_t k = 0; k < _subVectorSizes.size(); k++) {
      if(((1 / _residualSum[k])/_setWeights[k] > 10) || ((1 / _residualSum[k])/_setWeights[k] < 0.1)){
        resetWeight = 1;
        PRECICE_INFO("Resetting weights due to difference to previous weights in subvector: " << k);
      }
    }
    //if (((1 / _residualSum[0])/firstWeight > 10) || ((1 / _residualSum[0])/firstWeight < 0.1) || ((1 / _residualSum[2])/secondWeight > 10) || ((1 / _residualSum[2])/secondWeight < 0.1)){
    //    resetWeight = 1;
    //    PRECICE_INFO("Resetting weights due to difference to previous weights");
    //}
    for (size_t k = 0; k < _subVectorSizes.size(); k++) {
      //if (not math::equals(_residualSum[k], 0.0) ) {
        if (tStepPrecon < 2 || resetWeight == 1){
          for (size_t i = 0; i < _subVectorSizes[k]; i++) {
            _weights[i + offset]    = 1 / _residualSum[k];
            _invWeights[i + offset] = _residualSum[k];
          }
          PRECICE_DEBUG("preconditioner scaling factor[" << k << "] = " << 1 / _residualSum[k]);
        
          _setWeights[k] = 1 / _residualSum[k]; 
          //secondWeight = 1 / _residualSum[2];
        
          _requireNewQR = true;
          _updatedWeights = true;
          } 
        //}
      normWeights[k] = 1 / _residualSum[k];
      PRECICE_INFO("Actual Norm of weights: " << _weights[1 + offset]);
      PRECICE_INFO("Predicted Norm of weights: " << normWeights[k]);
      offset += _subVectorSizes[k];
      //PRECICE_INFO("Norm of weights: first " << firstWeight << " and Second: " << secondWeight);
    }
    resetWeight = 0;

  } else {
    tStepPrecon++;
    for (size_t k = 0; k < _subVectorSizes.size(); k++) {
      _residualSum[k] = 0.0;
    }
  }
}

} // namespace impl
} // namespace acceleration
} // namespace precice
