package org.smurve.deeplearning.optimizers

import breeze.linalg.DenseMatrix
import org.smurve.deeplearning.stats.NNStats
import org.smurve.deeplearning.{DM, DV}

/**
  * Created by wgiersche on 26.04.17.
  */
trait Optimizer {

  def nextStep ( gt: DM ) : DM

  /**
    * Convenience wrapper for DenseVectors
    * @param gt: The gradient
    * @return the next delta to update the given weights
    */
  def nextStep ( gt: DV ) : DV = {
    nextStep ( DenseMatrix(gt)).apply(0,::).t
  }
}
