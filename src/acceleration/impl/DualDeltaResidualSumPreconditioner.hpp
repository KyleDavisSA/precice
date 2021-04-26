#pragma once

#include <Eigen/Core>
#include <stddef.h>
#include <string>
#include <vector>
#include "acceleration/impl/Preconditioner.hpp"
#include "logging/Logger.hpp"

namespace precice {
namespace acceleration {
namespace impl {

/**
 * @brief Preconditioner that uses the residuals of all iterations of the current timestep summed up to scale the quasi-Newton system.
 * This is somewhat similar to what is done in the Marks and Luke paper.
 */
class DualDeltaResidualSumPreconditioner : public Preconditioner {
public:
  DualDeltaResidualSumPreconditioner(int maxNonConstTimesteps);
  /**
   * @brief Destructor, empty.
   */
  virtual ~DualDeltaResidualSumPreconditioner() {}

  virtual void initialize(std::vector<size_t> &svs);

  bool firstIter = true;

  int tStepPrecon = 1;
  int iterNumber = 0;       // Number of iterations inside the preconditioner
  Eigen::VectorXd normWeights;

private:
  /**
   * @brief Update the scaling after every FSI iteration.
   *
   * @param[in] timestepComplete True if this FSI iteration also completed a timestep
   */
  virtual void _update_(bool timestepComplete, const Eigen::VectorXd &oldValues, const Eigen::VectorXd &res, const Eigen::VectorXd &deltaRes);

  logging::Logger _log{"acceleration::DualDeltaResidualSumPreconditioner"};

  std::vector<double> _residualSum;
  std::vector<double> _setWeights;
  std::vector<double> _averageInitialWeight;
};

} // namespace impl
} // namespace acceleration
} // namespace precice
