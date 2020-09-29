#include "acceleration/BaseQNAcceleration.hpp"
#include <Eigen/Core>
#include <cmath>
#include <memory>
#include "acceleration/impl/Preconditioner.hpp"
#include "acceleration/impl/QRFactorization.hpp"
#include "com/Communication.hpp"
#include "com/SharedPointer.hpp"
#include "cplscheme/CouplingData.hpp"
#include "logging/LogMacros.hpp"
#include "mesh/Mesh.hpp"
#include "mesh/SharedPointer.hpp"
#include "utils/EigenHelperFunctions.hpp"
#include "utils/Event.hpp"
#include "utils/Helpers.hpp"
#include "utils/MasterSlave.hpp"
#include "utils/assertion.hpp"
#include "cplscheme/BaseCouplingScheme.hpp"
#include "cplscheme/CouplingScheme.hpp"

namespace precice {
namespace io {
class TXTReader;
class TXTWriter;
} // namespace io

extern bool syncMode;
namespace acceleration {

/* ----------------------------------------------------------------------------
 *     Constructor
 * ----------------------------------------------------------------------------
 */
BaseQNAcceleration::BaseQNAcceleration(
    double                  initialRelaxation,
    bool                    forceInitialRelaxation,
    int                     maxIterationsUsed,
    int                     timestepsReused,
    int                     filter,
    double                  singularityLimit,
    std::vector<int>        dataIDs,
    impl::PtrPreconditioner preconditioner)
    : _preconditioner(preconditioner),
      _initialRelaxation(initialRelaxation),
      _maxIterationsUsed(maxIterationsUsed),
      _timestepsReused(timestepsReused),
      _dataIDs(dataIDs),
      _forceInitialRelaxation(forceInitialRelaxation),
      _qrV(filter),
      _filter(filter),
      _singularityLimit(singularityLimit),
      _infostringstream(std::ostringstream::ate)
{
  PRECICE_CHECK((_initialRelaxation > 0.0) && (_initialRelaxation <= 1.0),
                "Initial relaxation factor for QN acceleration has to "
                    << "be larger than zero and smaller or equal than one. Current initial relaxation is: " << _initialRelaxation);
  PRECICE_CHECK(_maxIterationsUsed > 0,
                "Maximum number of iterations used in the quasi-Newton acceleration "
                    << "scheme has to be larger than zero. Current maximum reused iterations is: " << _maxIterationsUsed);
  PRECICE_CHECK(_timestepsReused >= 0,
                "Number of previous time windows to be reused for quasi-Newton acceleration has to be larger than or equal to zero. "
                    << "Current number of time windows reused is " << _timestepsReused);
}

/** ---------------------------------------------------------------------------------------------
 *         initialize()
 *
 * @brief: Initializes all the needed variables and data
 *  ---------------------------------------------------------------------------------------------
 */
void BaseQNAcceleration::initialize(
    DataMap &cplData)
{
  PRECICE_TRACE(cplData.size());
  checkDataIDs(cplData);

  /*
  std::stringstream sss;
  sss<<"debugOutput-rank-"<<utils::MasterSlave::getRank();
  _debugOut.open(sss.str(), std::ios_base::out);
  _debugOut << std::setprecision(16);

  Eigen::IOFormat CommaInitFmt(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", ", ", "", "", " << ", ";");

  _debugOut<<"initialization:\n";
  for (int id : _dataIDs) {
      const auto& values = *cplData[id]->values;
      const auto& oldValues = cplData[id]->oldValues.col(0);

      _debugOut<<"id: "<<id<<" dim: "<<cplData[id]->dimension<<"     values: "<<values.format(CommaInitFmt)<<'\n';
      _debugOut<<"id: "<<id<<" dim: "<<cplData[id]->dimension<<" old values: "<<oldValues.format(CommaInitFmt)<<'\n';
    }
  _debugOut<<"\n";
  */

  size_t              entries = 0;
  std::vector<size_t> subVectorSizes; //needed for preconditioner

  for (auto &elem : _dataIDs) {
    entries += cplData[elem]->values->size();
    subVectorSizes.push_back(cplData[elem]->values->size());
  }

  _matrixCols.push_front(0);
  _firstIteration = true;
  _firstTimeStep  = true;

  PRECICE_ASSERT(_oldXTilde.size() == 0);
  PRECICE_ASSERT(_oldResiduals.size() == 0);
  _oldXTilde    = Eigen::VectorXd::Zero(entries);
  _oldResiduals = Eigen::VectorXd::Zero(entries);
  _residuals    = Eigen::VectorXd::Zero(entries);
  _values       = Eigen::VectorXd::Zero(entries);
  _oldValues    = Eigen::VectorXd::Zero(entries);

  /**
   *  make dimensions public to all procs,
   *  last entry _dimOffsets[MasterSlave::getSize()] holds the global dimension, global,n
   */
  std::stringstream ss;
  if (utils::MasterSlave::isMaster() || utils::MasterSlave::isSlave()) {
    PRECICE_ASSERT(utils::MasterSlave::_communication.get() != NULL);
    PRECICE_ASSERT(utils::MasterSlave::_communication->isConnected());

    if (entries <= 0) {
      _hasNodesOnInterface = false;
    }

    /** provide vertex offset information for all processors
     *  mesh->getVertexOffsets() provides an array that stores the number of mesh vertices on each processor
     *  This information needs to be gathered for all meshes. To get the number of respective unknowns of a specific processor
     *  we need to multiply the number of vertices with the dimensionality of the vector-valued data for each coupling data.
     */
    _dimOffsets.resize(utils::MasterSlave::getSize() + 1);
    _dimOffsets[0] = 0;
    //for (auto & elem : _dataIDs) {
    //	std::cout<<" Offsets:(vertex) \n"<<cplData[elem]->mesh->getVertexOffsets()<<'\n';
    //}
    for (size_t i = 0; i < _dimOffsets.size() - 1; i++) {
      int accumulatedNumberOfUnknowns = 0;
      for (auto &elem : _dataIDs) {
        auto &offsets = cplData[elem]->mesh->getVertexOffsets();
        accumulatedNumberOfUnknowns += offsets[i] * cplData[elem]->dimension;
      }
      _dimOffsets[i + 1] = accumulatedNumberOfUnknowns;
    }
    PRECICE_DEBUG("Number of unknowns at the interface (global): " << _dimOffsets.back());
    if (utils::MasterSlave::isMaster()) {
      _infostringstream << "\n--------\n DOFs (global): " << _dimOffsets.back() << "\n offsets: " << _dimOffsets << '\n';
    }

    // test that the computed number of unknown per proc equals the number of entries actually present on that proc
    size_t unknowns = _dimOffsets[utils::MasterSlave::getRank() + 1] - _dimOffsets[utils::MasterSlave::getRank()];
    PRECICE_ASSERT(entries == unknowns, entries, unknowns);
  } else {
    _infostringstream << "\n--------\n DOFs (global): " << entries << '\n';
  }

  // set the number of global rows in the QRFactorization. This is essential for the correctness in master-slave mode!
  _qrV.setGlobalRows(getLSSystemRows());

  // Fetch secondary data IDs, to be relaxed with same coefficients from IQN-ILS
  for (DataMap::value_type &pair : cplData) {
    if (not utils::contained(pair.first, _dataIDs)) {
      _secondaryDataIDs.push_back(pair.first);
      int secondaryEntries            = pair.second->values->size();
      _secondaryResiduals[pair.first] = Eigen::VectorXd::Zero(secondaryEntries);
    }
  }

  // Append old value columns, if not done outside of acceleration already
  for (DataMap::value_type &pair : cplData) {
    int cols = pair.second->oldValues.cols();
    if (cols < 1) { // Add only, if not already done
      //PRECICE_ASSERT(pair.second->values->size() > 0, pair.first);
      utils::append(pair.second->oldValues, (Eigen::VectorXd) Eigen::VectorXd::Zero(pair.second->values->size()));
    }
  }

  _preconditioner->initialize(subVectorSizes);

  someConvergenceMeasure = _singularityLimit;
  if (someConvergenceMeasure > 0.8){
    if (_filter == Acceleration::QR1FILTER){
      _singularityLimit = 0.000001;
      upperLim = 0.002;
      lowerLim = 0.0000002;
    }
    if (_filter == Acceleration::QR2FILTER){
      _singularityLimit = 0.001;
      upperLim = 0.2;
      lowerLim = 0.001;
    }
  }
}

/** ---------------------------------------------------------------------------------------------
 *         updateDifferenceMatrices()
 *
 * @brief: computes the current residual and stores it, computes the differences and
 *         updates the difference matrices F and C.
 *  ---------------------------------------------------------------------------------------------
 */
void BaseQNAcceleration::updateDifferenceMatrices(
    DataMap &cplData)
{
  PRECICE_TRACE();

  // Compute current residual: vertex-data - oldData
  _residuals = _values;
  _residuals -= _oldValues;

  if (math::equals(utils::MasterSlave::l2norm(_residuals), 0.0)) {
    PRECICE_WARN("The coupling residual equals almost zero. There is maybe something wrong in your adapter. "
                 "Maybe you always write the same data or you call advance without "
                 "providing new data first or you do not use available read data. "
                 "Or you just converge much further than actually necessary.");
  }

  //if (_firstIteration && (_firstTimeStep || (_matrixCols.size() < 2))) {
  if (_firstIteration && (_firstTimeStep || _forceInitialRelaxation)) {
    // do nothing: constant relaxation
  } else {
    PRECICE_DEBUG("   Update Difference Matrices");
    if (not _firstIteration) {
      // Update matrices V, W with newest information

      PRECICE_ASSERT(_matrixV.cols() == _matrixW.cols(), _matrixV.cols(), _matrixW.cols());
      PRECICE_ASSERT(getLSSystemCols() <= _maxIterationsUsed, getLSSystemCols(), _maxIterationsUsed);

      if (2 * getLSSystemCols() >= getLSSystemRows())
        PRECICE_WARN(
            "The number of columns in the least squares system exceeded half the number of unknowns at the interface. "
            << "The system will probably become bad or ill-conditioned and the quasi-Newton acceleration may not "
            << "converge. Maybe the number of allowed columns (\"max-used-iterations\") should be limited.");

      Eigen::VectorXd deltaR = _residuals;
      deltaR -= _oldResiduals;

      Eigen::VectorXd deltaXTilde = _values;
      deltaXTilde -= _oldXTilde;

      PRECICE_CHECK(not math::equals(utils::MasterSlave::l2norm(deltaR), 0.0), "Attempting to add a zero vector to the quasi-Newton V matrix. This means that the residual "
                                                                               "in two consecutive iterations is identical. There is probably something wrong in your adapter. "
                                                                               "Maybe you always write the same (or only incremented) data or you call advance without "
                                                                               "providing  new data first.");


      bool columnLimitReached = getLSSystemCols() == _maxIterationsUsed;
      bool overdetermined     = getLSSystemCols() <= getLSSystemRows();
      if (not columnLimitReached && overdetermined) {

        utils::appendFront(_matrixV, deltaR);
        utils::appendFront(_matrixW, deltaXTilde);

        utils::appendFront(_matrixPseudoVSmall, deltaR);
        utils::appendFront(_matrixPseudoWSmall, deltaXTilde);

        // insert column deltaR = _residuals - _oldResiduals at pos. 0 (front) into the
        // QR decomposition and update decomposition

        //apply scaling here
        _preconditioner->apply(deltaR);
        _qrV.pushFront(deltaR);

        _matrixCols.front()++;
        _matrixColsSmall.front()++;
      } else {
        utils::shiftSetFirst(_matrixV, deltaR);
        utils::shiftSetFirst(_matrixW, deltaXTilde);

        utils::appendFront(_matrixPseudoVSmall, deltaR);
        utils::appendFront(_matrixPseudoWSmall, deltaXTilde);

        // inserts column deltaR at pos. 0 to the QR decomposition and deletes the last column
        // the QR decomposition of V is updated
        _preconditioner->apply(deltaR);
        _qrV.pushFront(deltaR);
        _qrV.popBack();

        _matrixCols.front()++;
        _matrixCols.back()--;
        _matrixColsSmall.front()++;
        _matrixColsSmall.back()--;
        if (_matrixCols.back() == 0) {
          _matrixCols.pop_back();
        }
        if (_matrixColsSmall.back() == 0) {
          _matrixColsSmall.pop_back();
        }
        _nbDropCols++;
      }
    }
    _oldResiduals = _residuals; // Store residuals
    _oldXTilde    = _values;    // Store x_tilde
  }
}

/** ---------------------------------------------------------------------------------------------
 *         performAcceleration()
 *
 * @brief: performs one iteration of the quasi Newton acceleration.
 *  ---------------------------------------------------------------------------------------------
 */
void BaseQNAcceleration::performAcceleration(
    DataMap &cplData)
{
  PRECICE_TRACE(_dataIDs.size(), cplData.size());

  utils::Event e("cpl.computeQuasiNewtonUpdate", precice::syncMode);

  PRECICE_ASSERT(_oldResiduals.size() == _oldXTilde.size(), _oldResiduals.size(), _oldXTilde.size());
  PRECICE_ASSERT(_values.size() == _oldXTilde.size(), _values.size(), _oldXTilde.size());
  PRECICE_ASSERT(_oldValues.size() == _oldXTilde.size(), _oldValues.size(), _oldXTilde.size());
  PRECICE_ASSERT(_residuals.size() == _oldXTilde.size(), _residuals.size(), _oldXTilde.size());

  /*
  Eigen::IOFormat CommaInitFmt(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", ", ", "", "", " << ", ";");
  _debugOut<<"iteration: "<<its<<" tStep: "<<tSteps<<"   cplData entry:\n";
  for (int id : _dataIDs) {
      const auto& values = *cplData[id]->values;
      const auto& oldValues = cplData[id]->oldValues.col(0);

      _debugOut<<"id: "<<id<<"     values: "<<values.format(CommaInitFmt)<<'\n';
      _debugOut<<"id: "<<id<<" old values: "<<oldValues.format(CommaInitFmt)<<'\n';
    }
  _debugOut<<"\n";
  */

  // assume data structures associated with the LS system can be updated easily.

  // scale data values (and secondary data values)
  concatenateCouplingData(cplData);
  //individualNormCouplingData(cplData);
  double dataNorm = utils::MasterSlave::l2norm(_oldValues);
  PRECICE_INFO("Norm of data: " << dataNorm);

  /** update the difference matrices V,W  includes:
   * scaling of values
   * computation of residuals
   * appending the difference matrices
   */
  updateDifferenceMatrices(cplData);

  if (_firstIteration && (_firstTimeStep || _forceInitialRelaxation) ) {

    PRECICE_DEBUG("   Performing underrelaxation");
    _oldXTilde    = _values;    // Store x tilde
    _oldResiduals = _residuals; // Store current residual
    

    // Perform constant relaxation
    // with residual: x_new = x_old + omega * res
    _residuals *= _initialRelaxation;
    _residuals += _oldValues;
    _values = _residuals;

    computeUnderrelaxationSecondaryData(cplData);
  } else {
    PRECICE_DEBUG("   Performing quasi-Newton Step");

    // If the previous time step converged within one single iteration, nothing was added
    // to the LS system matrices and they need to be restored from the backup at time T-2
    if (not _firstTimeStep && (getLSSystemCols() < 1) && (_timestepsReused == 0) && not _forceInitialRelaxation) {
      PRECICE_DEBUG("   Last time step converged after one iteration. Need to restore the matrices from backup.");

      _matrixCols = _matrixColsBackup;
      _matrixV    = _matrixVBackup;
      _matrixW    = _matrixWBackup;

      // re-computation of QR decomposition from _matrixV = _matrixVBackup
      // this occurs very rarely, to be precise, it occurs only if the coupling terminates
      // after the first iteration and the matrix data from time step t-2 has to be used
      _preconditioner->apply(_matrixV);
      _qrV.reset(_matrixV, getLSSystemRows());
      _preconditioner->revert(_matrixV);
      _resetLS = true; // need to recompute _Wtil, Q, R (only for IMVJ efficient update)
    }
    //if (!_firstTimeStep)
    //  _singularityLimit = 0.001;

    Eigen::VectorXd xUpdate = Eigen::VectorXd::Zero(_residuals.size());

      _matrixPseudoV = _matrixV;
      _matrixPseudoW = _matrixW;
      _matrixPseudoCols =_matrixCols;
      _qrVSaved = _qrV;

    Eigen::VectorXd newValues;
    //_singularityLimit = 0.0001;

    Eigen::VectorXd oldXUpdate = Eigen::VectorXd::Zero(_residuals.size());
   
    for (int i = 0; i < 1; i++){

      PRECICE_INFO("Number of matrix Columns: " <<  _matrixV.cols());
      lastChange =  _matrixV.cols();
      //PRECICE_INFO("Norm of matrix V: " << utils::MasterSlave::l2norm(_matrixV));
      PRECICE_INFO("Norm of new Values: " << utils::MasterSlave::l2norm(_values));
      PRECICE_INFO("Norm of old Values: " << utils::MasterSlave::l2norm(_oldValues));
      PRECICE_INFO("Norm difference: " << utils::MasterSlave::l2norm(_values) - utils::MasterSlave::l2norm(_oldValues));

      //if ((utils::MasterSlave::l2norm(_values) / utils::MasterSlave::l2norm(_oldValues)) > 100){
      //  _singularityLimit *= 10;
      //}

      //if (iterationsCheckConstantConverging == 54){
      //  _singularityLimit = 0.99999;
      //}

     // _singularityLimit *= 2;

    /**
     *  === update and apply preconditioner ===
     *
     * The preconditioner is only applied to the matrix V and the columns that are inserted into the
     * QR-decomposition of V.
     */

    // Reset the matrices V, W and number of columns
    /*
    _matrixV = _matrixPseudoV;
    _matrixW = _matrixPseudoW;
    _matrixCols = _matrixPseudoCols;
    PRECICE_INFO("Number of matrix Columns: " << _matrixCols);
    //_qrV = _qrVSaved;

    _preconditioner->apply(_matrixV);
    _qrV.reset(_matrixV, getLSSystemRows());
    _preconditioner->revert(_matrixV);
    _resetLS = true; // need to recompute _Wtil, Q, R (only for IMVJ efficient update)
    */

    _preconditioner->update(false, _values, _residuals);
    // apply scaling to V, V' := P * V (only needed to reset the QR-dec of V)
    _preconditioner->apply(_matrixV);

    if (_preconditioner->requireNewQR()) {
      if (not(_filter == Acceleration::QR2FILTER)) { //for QR2 filter, there is no need to do this twice
        _qrV.reset(_matrixV, getLSSystemRows());
      }
      _preconditioner->newQRfulfilled();
    }

    if (_firstIteration) {
      _nbDelCols  = 0;
      _nbDropCols = 0;
    }

    // apply the configured filter to the LS system
    /*
        Here I can call applyFilter multiple times with different settings to get 
        different answers for "_values". This matrix is saved and each column can
        be compared to the final answer
    */ 
   
    applyFilter();
    
    PRECICE_INFO("Number of columns:" << _matrixV.cols() << " with filter limit: " << _singularityLimit);
    deletedCols += (lastChange - _matrixV.cols());
    deletedColsConstantConverging += (lastChange - _matrixV.cols());
    PRECICE_INFO("Deleted Columns in last 5 iterations: " << deletedCols);
    // revert scaling of V, in computeQNUpdate all data objects are unscaled.
    _preconditioner->revert(_matrixV);
    /**
     * compute quasi-Newton update
     * PRECONDITION: All objects are unscaled, except the matrices within the QR-dec of V.
     *               Thus, the pseudo inverse needs to be reverted before using it.
     */

    xUpdate = Eigen::VectorXd::Zero(_residuals.size());
    computeQNUpdate(cplData, xUpdate);

    //_singularityLimit *= 10;

    
    if (i > 0){
      if (utils::MasterSlave::l2norm(xUpdate) > utils::MasterSlave::l2norm(oldXUpdate)){
        xUpdate = oldXUpdate;
      }
    }

    oldXUpdate = xUpdate;

    }

    _values = _oldValues + xUpdate + _residuals; // = x^k + delta_x + r^k - q^k
    newValues = _oldValues + xUpdate + _residuals;
    utils::appendFront(_testingValues, newValues);
    
    storeResults(_singularityLimit, newValues);
    double _isConvergenceTest;
    double normFirst;

    PRECICE_INFO("Output of getNorm in convergence writer in QN in performAcceleration: " << someConvergenceMeasure);


    if (iterationsToChange > 0) {
      double _normDiffTest      = utils::MasterSlave::l2norm(_values - _oldValues);
      double _normTest          = utils::MasterSlave::l2norm(_values);
      _isConvergenceTest = _normDiffTest/_normTest;
      PRECICE_INFO("Testing convergence: " << _isConvergenceTest);
      //if (iterationsToChange == 1){
      //  _isConvOld = _isConvergenceTest;
      //} 
    }
    

   // This indicate when the `old` convergence value is saved to be compared to the `new` convergence value later.
    if (iterationsToChange == 1){
      _isConvOld = _isConvergenceTest;
      //_oldValuesTest
    }

    
    /*
      This compares the convergence rate 5 iterations later. If the convergence has not reduced by more than
      a factor of 10, the filter limit is adjusted depending on the number of columns left. The minimum
      number of columns needed and the maximum is a tunable parameter. Need to find a value that 
      is good for lots of scenarios.
    */

    

    /*
      Decision Tree Fitler Variant 1:
        Only changes filter if # deleted columns is =1 or =5
        More suited to larger changes in filter limit
    */
    /*
    double singularityChangeFactor = 10.0;
    if (iterationsToChange == 5 && someConvergenceMeasure > 0.8){
      PRECICE_INFO("Old convergence value: " << _isConvOld);
      PRECICE_INFO("New convergence value: " << _isConvergenceTest);
      if (_isConvergenceTest > 0.2*(_isConvOld)){
        if (_matrixV.cols() > 15 && deletedCols < 2){
          _singularityLimit *= singularityChangeFactor;
          PRECICE_INFO("Enough Columns, less than one deleted. Increasing filter limit.");
          //RECICE_INFO("New filter limit: " << _singularityLimit);
          
        }
        if (_matrixV.cols() > 15 && deletedCols > 4){
          _singularityLimit /= singularityChangeFactor;
          PRECICE_INFO("Enough Columns, more than four deleted. Decreasing filter limit.");
          //RECICE_INFO("New filter limit: " << _singularityLimit);
          
        }
        if (_matrixV.cols() < 10  && deletedCols > 2){
          _singularityLimit /= singularityChangeFactor;
        
          PRECICE_INFO("Less than 10 Columns, and deleting more than 2. Decreasing filter limit.");
        }
        if (_matrixV.cols() < 10  && deletedCols == 0){
          _singularityLimit *= singularityChangeFactor;
          
          PRECICE_INFO("Less than 10 Columns, none deleted. Increasing filter limit.");
        }
        //if (_singularityLimit > upperLim){
        //  _singularityLimit /= singularityChangeFactor;
        //  PRECICE_INFO("Filter limit too high. Reducing by a factor of 10. ");
        //}
        //if (_singularityLimit < lowerLim){
        //  _singularityLimit *= singularityChangeFactor;
        //  PRECICE_INFO("Filter limit too low. Increasing by a factor of 10. ");
        //}
      } else {
        if (iterationsCheckConstantConverging > 10){
          if (_matrixV.cols() > 15 && deletedColsConstantConverging < 2){
            _singularityLimit *= singularityChangeFactor;
            PRECICE_INFO("Converging over 10 iterations and enough columns. Deleted 1 column only. Increasing filter limit.");
          }
          if (_matrixV.cols() > 15 && deletedColsConstantConverging > 9){
            _singularityLimit /= singularityChangeFactor;
            PRECICE_INFO("Converging over 10 iterations and enough columns. Deleted lots of columns. Decreasing filter limit.");
          }
          if (_matrixV.cols() < 10 && deletedColsConstantConverging > 3){
            _singularityLimit /= singularityChangeFactor;
            PRECICE_INFO("Converging over 10 iterations but few columns. Deleted a few columns. Decreasing filter limit.");
          }
          //if (_singularityLimit > upperLim){
          //  _singularityLimit /= singularityChangeFactor;
          //  PRECICE_INFO("Filter limit too high. Reducing by a factor of 10. ");
          //}
          //if (_singularityLimit < lowerLim){
          //  _singularityLimit *= singularityChangeFactor;
          //  PRECICE_INFO("Filter limit too low. Increasing by a factor of 10. ");
          //}
          //iterationsCheckConstantConverging = 0;
          deletedColsConstantConverging = 0;
        }
      }
      //if (_matrixV.cols() > 15 && deletedCols > 4){
      //    _singularityLimit /= singularityChangeFactor;
      //    PRECICE_INFO("Enough Columns, can filter more. Too many deleted");
          //RECICE_INFO("New filter limit: " << _singularityLimit);
      //  }
      if (_singularityLimit > upperLim){
          _singularityLimit /= singularityChangeFactor;
          PRECICE_INFO("Filter limit too high. Reducing by a factor of " << singularityChangeFactor);
        }
        if (_singularityLimit < lowerLim){
            _singularityLimit *= singularityChangeFactor;
            PRECICE_INFO("Filter limit too low. Increasing by a factor of " << singularityChangeFactor);
          }
      iterationsToChange = 0;
      _isConvOld = _isConvergenceTest;
      deletedCols = 0;
    }
    PRECICE_INFO("New filter limit: " << _singularityLimit);

    */

   /*
      Decision Tree Fitler Variant 2:
        Decreases filter if 4 or above
        Increases filter if 2 or below
        More suited for small increments such as *2, not *10
    
    */
    double singularityChangeFactor = 2.0;
    if (iterationsToChange == 5 && someConvergenceMeasure > 0.8){
      PRECICE_INFO("Old convergence value: " << _isConvOld);
      PRECICE_INFO("New convergence value: " << _isConvergenceTest);
      if (_isConvergenceTest > 0.2*(_isConvOld)){
        if (_matrixV.cols() > 15 && deletedCols < 3){
          _singularityLimit *= singularityChangeFactor;
          PRECICE_INFO("Enough Columns, less than one deleted. Increasing filter limit.");
          //RECICE_INFO("New filter limit: " << _singularityLimit);
          
        }
        if (_matrixV.cols() > 15 && deletedCols > 3){
          _singularityLimit /= singularityChangeFactor;
          PRECICE_INFO("Enough Columns, more than four deleted. Decreasing filter limit.");
          //RECICE_INFO("New filter limit: " << _singularityLimit);
          
        }
        if (_matrixV.cols() < 10  && deletedCols > 2){
          _singularityLimit /= singularityChangeFactor;
        
          PRECICE_INFO("Less than 10 Columns, and deleting more than 2. Decreasing filter limit.");
        }
        if (_matrixV.cols() < 10  && deletedCols == 0){
          _singularityLimit *= singularityChangeFactor;
          
          PRECICE_INFO("Less than 10 Columns, none deleted. Increasing filter limit.");
        }
        //if (_singularityLimit > upperLim){
        //  _singularityLimit /= singularityChangeFactor;
        //  PRECICE_INFO("Filter limit too high. Reducing by a factor of 10. ");
        //}
        //if (_singularityLimit < lowerLim){
        //  _singularityLimit *= singularityChangeFactor;
        //  PRECICE_INFO("Filter limit too low. Increasing by a factor of 10. ");
        //}
      } else {
        if (iterationsCheckConstantConverging > 10){
          if (_matrixV.cols() > 15 && deletedColsConstantConverging < 2){
            _singularityLimit *= singularityChangeFactor;
            PRECICE_INFO("Converging over 10 iterations and enough columns. Deleted 1 column only. Increasing filter limit.");
          }
          if (_matrixV.cols() > 15 && deletedColsConstantConverging > 8){
            _singularityLimit /= singularityChangeFactor;
            PRECICE_INFO("Converging over 10 iterations and enough columns. Deleted lots of columns. Decreasing filter limit.");
          }
          if (_matrixV.cols() < 10 && deletedColsConstantConverging > 3){
            _singularityLimit /= singularityChangeFactor;
            PRECICE_INFO("Converging over 10 iterations but few columns. Deleted a few columns. Decreasing filter limit.");
          }
          //if (_singularityLimit > upperLim){
          //  _singularityLimit /= singularityChangeFactor;
          //  PRECICE_INFO("Filter limit too high. Reducing by a factor of 10. ");
          //}
          //if (_singularityLimit < lowerLim){
          //  _singularityLimit *= singularityChangeFactor;
          //  PRECICE_INFO("Filter limit too low. Increasing by a factor of 10. ");
          //}
          //iterationsCheckConstantConverging = 0;
          deletedColsConstantConverging = 0;
        }
      }
      //if (_matrixV.cols() > 15 && deletedCols > 4){
      //    _singularityLimit /= singularityChangeFactor;
      //    PRECICE_INFO("Enough Columns, can filter more. Too many deleted");
          //RECICE_INFO("New filter limit: " << _singularityLimit);
      //  }
      if (_singularityLimit > upperLim){
          _singularityLimit /= singularityChangeFactor;
          PRECICE_INFO("Filter limit too high. Reducing by a factor of " << singularityChangeFactor);
        }
        if (_singularityLimit < lowerLim){
            _singularityLimit *= singularityChangeFactor;
            PRECICE_INFO("Filter limit too low. Increasing by a factor of " << singularityChangeFactor);
          }
      iterationsToChange = 0;
      _isConvOld = _isConvergenceTest;
      deletedCols = 0;
    }
    PRECICE_INFO("New filter limit: " << _singularityLimit);

  
        
  ///PRECICE_INFO("iterationsToChange: " << iterationsToChange);

    Eigen::VectorXd convTest;
    convTest.resize(_testingValues.rows(), 1);
    /*if (iterationsToChange == 45) {
      for (int i = 0; i < _testingValues.cols(); i++){
        for (int j = 0; j < _testingValues.rows(); j++) {
          convTest(j,0) = _testingValues(j,i);
        }

          double _normDiffTest      = utils::MasterSlave::l2norm(convTest - _oldValuesTest);
          double _normTest          = utils::MasterSlave::l2norm(convTest);
          double _isConvergenceTest = _normDiffTest/_normTest;
          PRECICE_INFO("Testing convergence: " << _isConvergenceTest);
        
      }
    }
    */
    
    //if (iterationsToChange == 65)
    //_oldValuesTest = _values;

    // pending deletion: delete old V, W matrices if timestepsReused = 0
    // those were only needed for the first iteration (instead of underrelax.)
    if (_firstIteration && _timestepsReused == 0 && not _forceInitialRelaxation) {
      // save current matrix data in case the coupling for the next time step will terminate
      // after the first iteration (no new data, i.e., V = W = 0)
      if (getLSSystemCols() > 0) {
        _matrixColsBackup = _matrixCols;
        _matrixVBackup    = _matrixV;
        _matrixWBackup    = _matrixW;
      }
      // if no time steps reused, the matrix data needs to be cleared as it was only needed for the
      // QN-step in the first iteration (idea: rather perform QN-step with information from last converged
      // time step instead of doing a underrelaxation)
      if (not _firstTimeStep) {
        _matrixV.resize(0, 0);
        _matrixW.resize(0, 0);
        _matrixCols.clear();
        _matrixCols.push_front(0); // vital after clear()
        _qrV.reset();
        // set the number of global rows in the QRFactorization. This is essential for the correctness in master-slave mode!
        _qrV.setGlobalRows(getLSSystemRows());
        _resetLS = true; // need to recompute _Wtil, Q, R (only for IMVJ efficient update)
      }
    }

    if (std::isnan(utils::MasterSlave::l2norm(xUpdate))) {
      PRECICE_ERROR("The quasi-Newton update contains NaN values. This means that the quasi-Newton acceleration failed to converge. "
                    "When writing your own adapter this could indicate that you give wrong information to preCICE, such as identical "
                    "data in succeeding iterations. Or you do not properly save and reload checkpoints. "
                    "If you give the correct data this could also mean that the coupled problem is too hard to solve. Try to use a QR "
                    "filter or increase its threshold (larger epsilon).");
    }
  }

  //if (iterationsCheckConstantConverging == 54){
  //      _singularityLimit = someConvergenceMeasure;
  //}

  iterationsToChange++;
  iterationsCheckConstantConverging++;

  splitCouplingData(cplData);

  /*
  _debugOut<<"finished update: \n";
  for (int id : _dataIDs) {
      const auto& values = *cplData[id]->values;
      const auto& oldValues = cplData[id]->oldValues.col(0);

      _debugOut<<"id: "<<id<<"norm: "<<values.norm()<<"     values: "<<values.format(CommaInitFmt)<<'\n';
      _debugOut<<"id: "<<id<<"norm: "<<oldValues.norm()<<" old values: "<<oldValues.format(CommaInitFmt)<<'\n';
    }
  _debugOut<<"\n";
  */

  // number of iterations (usually equals number of columns in LS-system)
  its++;
  _firstIteration = false;
}

void BaseQNAcceleration::newConvMeasure(double newConvMeasure)
{
  someConvergenceMeasure = newConvMeasure;
  PRECICE_INFO("Output of getNorm in convergence writer in QN: " << someConvergenceMeasure);
}

void BaseQNAcceleration::applyFilter()
{
  PRECICE_TRACE(_filter);
  int remaining;

  if (_filter == Acceleration::NOFILTER) {
    // do nothing
  } else {
    // do: filtering of least-squares system to maintain good conditioning
    std::vector<int> delIndices(0);
    _qrV.applyFilter(_singularityLimit, delIndices, _matrixV);
    // start with largest index (as V,W matrices are shrinked and shifted
    
    PRECICE_INFO("Columns in applyFilter: " << _qrV.cols());
    remaining = _qrV.cols()/2;
    //if (_qrV.cols()/2 < 10){
    //  remaining = 10;
   // }

    //if (_qrV.cols() > 10){
      
      for (int i = delIndices.size() - 1; i >= 0; i--) {
        //if (remaining < 10)
        //  break;
        PRECICE_INFO(" Filter: removing column with index " << delIndices[i] << " in iteration " << its << " of time step: " << tSteps);
        removeMatrixColumn(delIndices[i]);
        
        //remaining--;
      
  }
    PRECICE_ASSERT(_matrixV.cols() == _qrV.cols(), _matrixV.cols(), _qrV.cols());
  }
}

void BaseQNAcceleration::parameterTuning()
{
  PRECICE_TRACE(_filter);

  if (_filter == Acceleration::NOFILTER) {
    // do nothing
  } else {
   // _singularityLimit /= 10;
  }
  //if (_singularityLimit < 0.00000001){
  //  _singularityLimit = 0.00000001;
  //}

  //PRECICE_INFO("New filter limit: " << _singularityLimit);
}

void BaseQNAcceleration::resetIterationsToChange(int someInt)
{
  iterationsToChange = 0;
}

void BaseQNAcceleration::storeResults(double limit, Eigen::VectorXd newValues)
{
  PRECICE_INFO("New values for storing results with limit: " << limit);
  for (int i = 0; i < 10; i++){
    PRECICE_INFO(" " << newValues[i]);
  }
  
}

void BaseQNAcceleration::concatenateCouplingData(
    DataMap &cplData)
{
  PRECICE_TRACE();

  int offset = 0;
  for (int id : _dataIDs) {
    int         size      = cplData[id]->values->size();
    auto &      values    = *cplData[id]->values;
    const auto &oldValues = cplData[id]->oldValues.col(0);
    double inputNorm = utils::MasterSlave::l2norm(values);
    PRECICE_INFO("Input Norm of data ID: " << id << " - with norm: " << inputNorm);
    for (int i = 0; i < size; i++) {
      _values(i + offset)    = values(i);
      _oldValues(i + offset) = oldValues(i);
    }
    offset += size;
  }
}

/*
void BaseQNAcceleration::individualNormCouplingData(
    DataMap &cplData)
{
  PRECICE_TRACE();

  int offset = 0;
  for (int id : _dataIDs) {
    int         size      = cplData[id]->values->size();
    auto &      values    = *cplData[id]->values;
    const auto &oldValues = cplData[id]->oldValues.col(0);
    double inputNorm = utils::MasterSlave::l2norm(values);
    PRECICE_INFO("Input Norm of data ID: " << id << " - with norm: " << inputNorm);
  }
}
*/



void BaseQNAcceleration::splitCouplingData(
    DataMap &cplData)
{
  PRECICE_TRACE();

  int offset = 0;
  for (int id : _dataIDs) {
    int   size       = cplData[id]->values->size();
    auto &valuesPart = *(cplData[id]->values);
    //Eigen::VectorXd& oldValuesPart = cplData[id]->oldValues.col(0);
    cplData[id]->oldValues.col(0) = _oldValues.segment(offset, size); /// @todo: check if this is correct
    for (int i = 0; i < size; i++) {
      valuesPart(i) = _values(i + offset);
      //oldValuesPart(i) = _oldValues(i + offset);
    }
    offset += size;
  }
}

/** ---------------------------------------------------------------------------------------------
 *         iterationsConverged()
 *
 * @brief: Is called when the convergence criterion for the coupling is fullfilled and finalizes
 *         the quasi Newton acceleration. Stores new differences in F and C, clears or
 *         updates F and C according to the number of reused time steps
 *  ---------------------------------------------------------------------------------------------
 */
void BaseQNAcceleration::iterationsConverged(
    DataMap &cplData)
{
  PRECICE_TRACE();

  if (utils::MasterSlave::isMaster() || (not utils::MasterSlave::isMaster() && not utils::MasterSlave::isSlave()))
    _infostringstream << "# time step " << tSteps << " converged #\n iterations: " << its
                      << "\n used cols: " << getLSSystemCols() << "\n del cols: " << _nbDelCols << '\n';

  its = 0;
  tSteps++;

  // the most recent differences for the V, W matrices have not been added so far
  // this has to be done in iterations converged, as PP won't be called any more if
  // convergence was achieved
  concatenateCouplingData(cplData);
  updateDifferenceMatrices(cplData);

  if (not _matrixCols.empty() && _matrixCols.front() == 0) { // Did only one iteration
    _matrixCols.pop_front();
  }

#ifndef NDEBUG
  std::ostringstream stream;
  stream << "Matrix column counters: ";
  for (int cols : _matrixCols) {
    stream << cols << ", ";
  }
  PRECICE_DEBUG(stream.str());
#endif // Debug

  // doing specialized stuff for the corresponding acceleration scheme after
  // convergence of iteration i.e.:
  // - analogously to the V,W matrices, remove columns from matrices for secondary data
  // - save the old Jacobian matrix
  specializedIterationsConverged(cplData);

  // if we already have convergence in the first iteration of the first timestep
  // we need to do underrelax in the first iteration of the second timesteps
  // so "_firstTimeStep" is slightly misused, but still the best way to understand
  // the concept
  if (not _firstIteration)
    _firstTimeStep = false;

  // update preconditioner depending on residuals or values (must be after specialized iterations converged --> IMVJ)
  _preconditioner->update(true, _values, _residuals);

  if (_timestepsReused == 0) {
    if (_forceInitialRelaxation) {
      _matrixV.resize(0, 0);
      _matrixW.resize(0, 0);
      _qrV.reset();
      // set the number of global rows in the QRFactorization. This is essential for the correctness in master-slave mode!
      _qrV.setGlobalRows(getLSSystemRows());
      _matrixCols.clear(); // _matrixCols.push_front() at the end of the method.
    } else {
      /**
       * pending deletion (after first iteration of next time step
       * Using the matrices from the old time step for the first iteration
       * is better than doing underrelaxation as first iteration of every time step
       */
    }
  } else if ((int) _matrixCols.size() > _timestepsReused) {
    int toRemove = _matrixCols.back();
    _nbDropCols += toRemove;
    PRECICE_ASSERT(toRemove > 0, toRemove);
    PRECICE_DEBUG("Removing " << toRemove << " cols from least-squares system with " << getLSSystemCols() << " cols");
    PRECICE_ASSERT(_matrixV.cols() == _matrixW.cols(), _matrixV.cols(), _matrixW.cols());
    PRECICE_ASSERT(getLSSystemCols() > toRemove, getLSSystemCols(), toRemove);

    // remove columns
    for (int i = 0; i < toRemove; i++) {
      utils::removeColumnFromMatrix(_matrixV, _matrixV.cols() - 1);
      utils::removeColumnFromMatrix(_matrixW, _matrixW.cols() - 1);
      // also remove the corresponding columns from the dynamic QR-descomposition of _matrixV
      _qrV.popBack();
    }
    _matrixCols.pop_back();
  }

  _matrixCols.push_front(0);
  _firstIteration = true;
}

/** ---------------------------------------------------------------------------------------------
 *         removeMatrixColumn()
 *
 * @brief: removes a column from the least squares system, i. e., from the matrices F and C
 *  ---------------------------------------------------------------------------------------------
 */
void BaseQNAcceleration::removeMatrixColumn(
    int columnIndex)
{
  PRECICE_INFO("RemoveMat columns: " << _matrixV.cols() << " - and columnIndex: " << columnIndex);
  PRECICE_TRACE(columnIndex, _matrixV.cols());

  _nbDelCols++;

  PRECICE_ASSERT(_matrixV.cols() > 1);
  
  utils::removeColumnFromMatrix(_matrixV, columnIndex);
  utils::removeColumnFromMatrix(_matrixW, columnIndex);

  // Reduce column count
  std::deque<int>::iterator iter = _matrixCols.begin();
  int                       cols = 0;
  while (iter != _matrixCols.end()) {
    cols += *iter;
    if (cols > columnIndex) {
      PRECICE_ASSERT(*iter > 0);
      *iter -= 1;
      if (*iter == 0) {
        _matrixCols.erase(iter);
      }
      break;
    }
    iter++;
  }
}

void BaseQNAcceleration::exportState(
    io::TXTWriter &writer)
{
}

void BaseQNAcceleration::importState(
    io::TXTReader &reader)
{
}

int BaseQNAcceleration::getDeletedColumns() const
{
  return _nbDelCols;
}

int BaseQNAcceleration::getDroppedColumns() const
{
  return _nbDropCols;
}

int BaseQNAcceleration::getLSSystemCols() const
{
  int cols = 0;
  for (int col : _matrixCols) {
    cols += col;
  }
  if (_hasNodesOnInterface) {
    PRECICE_ASSERT(cols == _matrixV.cols(), cols, _matrixV.cols(), _matrixCols, _qrV.cols());
    PRECICE_ASSERT(cols == _matrixW.cols(), cols, _matrixW.cols());
  }

  return cols;
}

int BaseQNAcceleration::getLSSystemRows()
{
  if (utils::MasterSlave::isMaster() || utils::MasterSlave::isSlave()) {
    return _dimOffsets.back();
  }
  return _residuals.size();
}

void BaseQNAcceleration::writeInfo(
    std::string s, bool allProcs)
{
  if (not utils::MasterSlave::isMaster() && not utils::MasterSlave::isSlave()) {
    // serial acceleration mode
    _infostringstream << s;

    // parallel acceleration, master-slave mode
  } else {
    if (not allProcs) {
      if (utils::MasterSlave::isMaster())
        _infostringstream << s;
    } else {
      _infostringstream << s;
    }
  }
  _infostringstream << std::flush;
}
} // namespace acceleration
} // namespace precice
