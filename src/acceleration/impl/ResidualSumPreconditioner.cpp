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
    PRECICE_INFO("Sum of res-sum preconditioner: " << sum);
    PRECICE_CHECK(not math::equals(sum, 0.0), "All residual sub-vectors in the residual-sum preconditioner are numerically zero. "
                                              "Your simulation probably got unstable, e.g. produces NAN values.");
    
    for (size_t k = 0; k < _subVectorSizes.size(); k++) {
      _residualSum[k] += norms[k] / sum;
      PRECICE_INFO("Norm of res-sum: " << norms[k]);
      PRECICE_INFO("residualSum of res-sum: " << _residualSum[k]);
      PRECICE_CHECK(not math::equals(_residualSum[k], 0.0), "A sub-vector in the residual-sum preconditioner became numerically zero. "
                                                            "Thus, the preconditioner is no longer stable. Please try the value preconditioner instead.");
    }

    offset = 0;
    int resetWeight = 0;
    normWeights.resize(_subVectorSizes.size());
    if (((1 / _residualSum[0])/maxWeight > 10) || ((1 / _residualSum[0])/maxWeight < 0.1) || ((1 / _residualSum[1])/minWeight > 10) || ((1 / _residualSum[1])/minWeight < 0.1)){
        resetWeight = 1;
    }
    //resetWeight = 1;
    for (size_t k = 0; k < _subVectorSizes.size(); k++) {
      //if (k == 0){
      
      //}
      /*if (k == 1){
        if (((1 / _residualSum[k])/minWeight > 5) || ((1 / _residualSum[k])/minWeight < 0.2)){
          resetWeight = 1;
        }
      }*/
      if (tStepPrecon < 2 || resetWeight == 1){
        for (size_t i = 0; i < _subVectorSizes[k]; i++) {
          _weights[i + offset]    = 1 / _residualSum[k];
          _invWeights[i + offset] = _residualSum[k];
        }
          maxWeight = 1 / _residualSum[0]; 
          minWeight = 1 / _residualSum[1];
        
        _requireNewQR = true;
        _updatedWeights = true;
      } 
      
      normWeights[k] = 1 / _residualSum[k];
      PRECICE_INFO("Actual Norm of weights: " << _weights[1 + offset]);
      PRECICE_INFO("Predicted Norm of weights: " << normWeights[k]);
      offset += _subVectorSizes[k];
      if (k == 0){
        //if (normWeights[k] > maxWeight)
          //maxWeight = normWeights[k];
        //if (normWeights[k] < minWeight && normWeights[k] > 1)
          //minWeight = normWeights[k];
      }
      PRECICE_INFO("Norm of weights: Min " << minWeight << " and Max: " << maxWeight);
    }
    resetWeight = 0;
    
    //PRECICE_INFO("Norm of weights: " << utils::MasterSlave::l2norm(normWeights));

    //_requireNewQR = true;
    //_requireNewQR = false;
    //}
  }else{

    //double rationOne = maxWeight/minWeight;
    //double rationTwo = (1 / _residualSum[0])/(1 / _residualSum[1]);

    //if(tStepPrecon < 2){
    //  maxWeight = 1 / _residualSum[0];
    //  minWeight = 1 / _residualSum[1];
    //}
    /*if(((maxWeight/(1 / _residualSum[0])) > 5 || (maxWeight/(1 / _residualSum[0])) < 0.2) || ((minWeight/(1 / _residualSum[1])) > 5 || (minWeight/(1 / _residualSum[1])) < 0.2)){
    //if(((rationOne/rationTwo < 0.1) || (rationOne/rationTwo > 10)) && (tStepPrecon > 7)){
      PRECICE_INFO("Updated weights MAX during runtime. Will need to reset SVD with ration: " << rationOne/rationTwo);
      
      int offset = 0;
      for (size_t k = 0; k < _subVectorSizes.size(); k++) {
        for (size_t i = 0; i < _subVectorSizes[k]; i++) {
          //_residualSum[0] = 2;
          _weights[i + offset]    = 1 / _residualSum[k];
          _invWeights[i + offset] = _residualSum[k];
        }
        offset += _subVectorSizes[k];
      }
      
      maxWeight = 1 / _residualSum[0];
      minWeight = 1 / _residualSum[1];
      _updatedWeights = true;
      _requireNewQR = true;
    }
    */
    /*if(((minWeight/(1 / _residualSum[1])) > 5 || (minWeight/(1 / _residualSum[1])) < 0.2)){
      PRECICE_INFO("Updated weights MIN during runtime. Will need to reset SVD with ration: " << rationOne/rationTwo);
      
      int offset = 0;
      //for (size_t k = 0; k < _subVectorSizes.size(); k++) {
        offset += _subVectorSizes[0];
        for (size_t i = 0; i < _subVectorSizes[1]; i++) {
         // _residualSum[1] = 2;
          _weights[i + offset]    = 1 / _residualSum[1];
          _invWeights[i + offset] = _residualSum[1];
        }
      //}
      
      minWeight = 1 / _residualSum[1];
      _updatedWeights = true;
      _requireNewQR = true;
    }
    */
    //_updatedWeights = true;
    //_requireNewQR = true;
    tStepPrecon++;
    for (size_t k = 0; k < _subVectorSizes.size(); k++) {
      _residualSum[k] = 0.0;
    }
  }
}

} // namespace impl
} // namespace acceleration
} // namespace precice
